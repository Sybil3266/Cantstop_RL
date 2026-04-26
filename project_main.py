from Cantstop_Gamefile import Cantstop
import Cantstop_DQN
from Cantstop_ReplayMemory import Cantstop_ReplayMemory
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from matplotlib import pyplot as plt
import random
import math
import copy
import numpy as np
import os
import datetime
import Fixed_policy as fp
from collections import Counter
import warnings
import sys
import time
import torch.multiprocessing as mp

import pandas as pd

warnings.filterwarnings("error")

BATCH_SIZE = 128
GAMMA = 0.9
GAMMA_DICE = 0.6  # dice 네트워크용 낮은 gamma (확률적 노이즈 감소)
LR_DICE = None     # dice 전용 LR (None이면 LR과 동일)
EPS_START = 0.9
EPS_END = 0.01

EPS_END_D = 0.01
EPS_END_A = 0.01

EPS_DECAY = 500
TAU = 0.005
LR = 1e-4
REPLAY_LIMIT = 25000
UPDATE_PER_EPISODE = 20
EPI_LEN = 2000
STEP_LIMIT = 150
UPDATE_PER_STEP = 5
# Stop 마스킹 임계값. flag 수가 이 값 미만일 때 stop 행동을 마스킹 (실험용 변수)
#   0 => 마스킹 없음 (agent 자유 선택)
#   2 => flag<2 에서 stop 차단 (현재 기본)
STOP_MASK_FLAG_THRESHOLD = 2
ep = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#random.seed(6)
#학습 진행
def learn(use_schedule = False, min_lr = 1e-6, experiment_name = None,
          eval_every=0, eval_n_games=100, eval_opponents=(1, 2, 3)):
    """
    self-play 학습.
    eval_every > 0 이면 해당 episode 마다 fixed_policy 풀에 _quick_eval 호출,
    snapshot winrate 를 그래프로 저장 (mid-training collapse 감지).
    """
    global device
    global ep
    print("START MAIN")
    print(f"BATCH_SIZE : {BATCH_SIZE}")
    if eval_every > 0:
        print(f"PERIODIC EVAL: every {eval_every} ep, {eval_n_games} games per opponent {eval_opponents}")

    print(f"Device is {device}")
    # 채널 분리 구조: 각 semantic row 를 독립 채널로 사용
    input_channels_dice = 7   # player/opponent/turn progress, conquered, 3 possible actions row
    input_channels_action = 4 # player/opponent/turn progress, conquered
    input_width = 11
    output_dim_dice = 10
    output_dim_action = 3

    num_epoch = 200
    updatecnt = 0
    # 모델 초기화 (Conv1d 기반 채널 분리)
    target_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice).to(device)
    policy_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice).to(device)
    target_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action).to(device)
    policy_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action).to(device)

    optim_dice, schedule_dice = optim_schedule(policy_selectDice, min_lr, lr_override=LR_DICE)
    optim_action, schedule_action = optim_schedule(policy_selectAction, min_lr)

    replay_memory_dice = Cantstop_ReplayMemory(action_size = output_dim_dice, buffer_size = REPLAY_LIMIT, batch_size = BATCH_SIZE)
    replay_memory_action = Cantstop_ReplayMemory(action_size = output_dim_action, buffer_size = REPLAY_LIMIT, batch_size = BATCH_SIZE)

    game = Cantstop()

    avg_rewards_dice = []
    avg_rewards_action = []
    avg_step = []
    avg_loss_dice = []
    avg_loss_action = []
    eval_log = []   # [(episode, {opp: winrate}), ...]
    start_replay = False
    updated = False
    train_start_time = time.time()
    print("START EPISODE")
    for episode in range(EPI_LEN):
        ep += 1
        game.Learn_reset()
        game_step = STEP_LIMIT - 1
        episode_losses_dice = []
        episode_losses_action = []
        if episode % 10 == 0:
            elapsed = time.time() - train_start_time
            eps_per_sec = (episode / elapsed) if elapsed > 0 and episode > 0 else 0
            print(f"episode: {episode} | elapsed: {elapsed:.1f}s | ep/s: {eps_per_sec:.2f}")
        for step in range(STEP_LIMIT):
            rewards_dice = 0
            rewards_action = 0
            
            action_matrix, action_masking = game.Learn_dice()

            dicecomb_input_mat = game.Learn_getState(1)
            
            policy_selectDice.eval()
            with torch.no_grad():
                action = select_action(action_matrix, action_masking, episode, 1, policy_selectDice, dicecomb_input_mat)
            policy_selectDice.train()
            if action != 9:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, action_matrix[action])
            else:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, ())

            rewards_dice += get_rewards_dice
            done_flag_dice = 0 if result == -1 else 1
            replay_memory_dice.add(dicecomb_input_mat, action, get_rewards_dice, updated_state_dice, done_flag_dice)
            selectaction_input_mat = game.Learn_getState(2)

            if action != 9:
                # 🅲: 도메인 지식 기반 stop 마스킹 (실험용 STOP_MASK_FLAG_THRESHOLD 로 제어)
                #  - threshold=0 이면 마스킹 OFF (flag 수 무관)
                #  - threshold=2 이면 flag<2 에서 stop 차단
                if STOP_MASK_FLAG_THRESHOLD > 0 and len(game.flags) < STOP_MASK_FLAG_THRESHOLD:
                    action_masking = [1, 0, 0]   # continue만 허용
                else:
                    action_masking = [1, 1, 0]
            else:
                action_masking = [0, 0, 1]

            policy_selectAction.eval()
            with torch.no_grad():
                action = select_action(action_matrix, action_masking, episode, 2, policy_selectAction, selectaction_input_mat)
            
            if action_masking[2] == 1 and action != 2:
                print(f"\n error \n")
                return
            policy_selectAction.train()

            game_result, get_rewards_action, updated_state_action = game.update_state_action(action, get_rewards_dice)
            rewards_action += get_rewards_action
            done_flag_action = 0 if game_result == -1 else 1
            replay_memory_action.add(selectaction_input_mat, action, get_rewards_action, updated_state_action, done_flag_action)

            isError, isFinished = game.check_error()

            if isError:
                return -1

            if isFinished is True and game_result == -1:
                print(f"GAME Finished BUT NOT RETURN")
                return -1

            if game_result != -1:
                game_step = step
                break
            if start_replay is True: #and episode % UPDATE_PER_EPISODE == 0:
                if step % UPDATE_PER_STEP == 0:
                    profile = (updatecnt % num_epoch == 0)
                    l_d, l_a = replay_batch(replay_memory_dice, replay_memory_action, optim_dice, optim_action, target_selectDice, policy_selectDice, target_selectAction, policy_selectAction, profile)
                    episode_losses_dice.append(l_d)
                    episode_losses_action.append(l_a)
                    #updatecnt += 1
                    updated = True
            # if start_replay and step % UPDATE_PER_STEP == 0:
            #     profile = (updatecnt % num_epoch == 0)
            #     # 병렬 프로세스 생성 및 실행
            #     if len(processes) < mp.cpu_count():
            #         p = mp.Process(target=replay_batch, args=(replay_memory_dice, replay_memory_action, optim_dice, optim_action, target_selectDice, policy_selectDice, target_selectAction, policy_selectAction))
            #         p.start()
            #         processes.append(p)
            #     else:
            #         for p in processes:
            #             p.join()
            #         processes = []
            #         updatecnt += 1
            #         updated = True

        game_step += 1
        avg_step.append(game_step)
        avg_rewards_dice.append(float(rewards_dice) / float(game_step))
        avg_rewards_action.append(float(rewards_action) / float(game_step))
        if episode_losses_dice:
            avg_loss_dice.append(np.mean(episode_losses_dice))
            avg_loss_action.append(np.mean(episode_losses_action))

        if replay_memory_dice.__len__() >= REPLAY_LIMIT:
            start_replay = True

        if use_schedule is True and updated is True:
            schedule_dice.step()
            schedule_action.step()

        if episode % UPDATE_PER_EPISODE == 0:
            target_selectDice = copy.deepcopy(policy_selectDice)
            target_selectAction = copy.deepcopy(policy_selectAction)

        # ----- Periodic eval (collapse 감지) -----
        if eval_every > 0 and episode > 0 and episode % eval_every == 0:
            snap = {}
            for opp in eval_opponents:
                w, l, d = _quick_eval(policy_selectDice, policy_selectAction, opp, n_games=eval_n_games)
                wr = w / (w + l) if (w + l) > 0 else 0.5
                snap[opp] = wr
            eval_log.append((episode, snap))
            opp_names = {1: 'Rnd', 2: 'H@3', 3: 'SoB'}
            snap_str = " | ".join(f"{opp_names.get(o, str(o))}={wr:.3f}" for o, wr in snap.items())
            print(f"  [EVAL @ep{episode}]  {snap_str}")

    total_time = time.time() - train_start_time
    print(f"\n=== Training Complete ===")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Episodes: {EPI_LEN} | Avg ep/s: {EPI_LEN/total_time:.2f}")
    print(f"Final avg loss - dice: {avg_loss_dice[-1]:.4f}, action: {avg_loss_action[-1]:.4f}" if avg_loss_dice else "No replay updates")
    if eval_log:
        print("\n=== Periodic eval snapshots ===")
        for ep_n, snap in eval_log:
            opp_names = {1: 'Rnd', 2: 'H@3', 3: 'SoB'}
            snap_str = " | ".join(f"{opp_names.get(o, str(o))}={wr:.3f}" for o, wr in snap.items())
            print(f"  ep{ep_n}: {snap_str}")

    save_model(policy_selectDice, policy_selectAction, avg_rewards_dice, avg_rewards_action, avg_step, None, None, None, None, use_schedule, min_lr, avg_loss_dice, avg_loss_action, experiment_name=experiment_name)
    if eval_log:
        _save_eval_snapshots(eval_log, experiment_name)


# ======================================================================
# Plan A: 고정 룰베이스 상대 학습 (self-play collapse 우회)
# ----------------------------------------------------------------------
# 기존 learn() 은 양쪽 모두 policy network 로 학습 → 둘 다 stop 편향으로
# 수렴해버려 승패 신호가 희석되는 self-play collapse 발생.
# 여기선 상대를 Hold_At_Three / Score_or_Bust 같은 고정 정책으로 두고
# 에이전트 한쪽만 학습 → "일찍 stop 하면 진다" 라는 확실한 신호 확보.
#
# 주요 차이점:
#   1. agent_player 지정 (기본 1). 반대쪽은 fixed_policy_num 의 call_policy 로 선택
#   2. 상대 턴에는 replay_memory.add() 생략 — 에이전트 학습 데이터 아님
#   3. 상대 턴에도 game.update_state_* 은 호출해 게임 진행은 지속
#   4. win_count 추적 — 에피소드 단위 승률을 직접 집계
# ======================================================================
def learn_vs_fixed(fixed_policy_num=2, use_schedule=False, min_lr=1e-6, experiment_name=None, agent_player=1):
    global device
    global ep
    print("START MAIN (vs fixed opponent)")
    print(f"fixed_policy_num = {fixed_policy_num}  (2=Hold_At_Three, 3=Score_or_Bust)")
    print(f"agent_player = {agent_player}")
    print(f"BATCH_SIZE : {BATCH_SIZE}")
    print(f"Device is {device}")

    input_channels_dice = 7
    input_channels_action = 4
    input_width = 11
    output_dim_dice = 10
    output_dim_action = 3

    num_epoch = 200
    updatecnt = 0

    target_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice).to(device)
    policy_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice).to(device)
    target_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action).to(device)
    policy_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action).to(device)

    optim_dice, schedule_dice = optim_schedule(policy_selectDice, min_lr, lr_override=LR_DICE)
    optim_action, schedule_action = optim_schedule(policy_selectAction, min_lr)

    replay_memory_dice = Cantstop_ReplayMemory(action_size=output_dim_dice, buffer_size=REPLAY_LIMIT, batch_size=BATCH_SIZE)
    replay_memory_action = Cantstop_ReplayMemory(action_size=output_dim_action, buffer_size=REPLAY_LIMIT, batch_size=BATCH_SIZE)

    game = Cantstop()

    avg_rewards_dice = []
    avg_rewards_action = []
    avg_step = []
    avg_loss_dice = []
    avg_loss_action = []
    # Plan A 전용 로그: 에피소드 단위 승률 추적
    win_history = []   # 1=agent win, 0=lose, -1=draw/timeout
    rolling_winrate = []

    start_replay = False
    updated = False
    train_start_time = time.time()
    print("START EPISODE")

    for episode in range(EPI_LEN):
        ep += 1
        game.Learn_reset()
        game_step = STEP_LIMIT - 1
        episode_losses_dice = []
        episode_losses_action = []
        agent_rewards_dice = 0
        agent_rewards_action = 0
        final_result = -1   # -1: ongoing / draw, 1: someone won (본 episode)
        winner_is_agent = False

        if episode % 50 == 0:
            elapsed = time.time() - train_start_time
            eps_per_sec = (episode / elapsed) if elapsed > 0 and episode > 0 else 0
            recent_wr = np.mean(win_history[-100:]) if len(win_history) >= 10 else float('nan')
            print(f"episode: {episode} | elapsed: {elapsed:.1f}s | ep/s: {eps_per_sec:.2f} | recent100 winrate: {recent_wr:.3f}")

        for step in range(STEP_LIMIT):
            cur_player = game.player
            is_agent_turn = (cur_player == agent_player)

            # ---------------- Dice phase ----------------
            action_matrix, action_masking = game.Learn_dice()
            dicecomb_input_mat = game.Learn_getState(1)

            if is_agent_turn:
                policy_selectDice.eval()
                with torch.no_grad():
                    action = select_action(action_matrix, action_masking, episode, 1, policy_selectDice, dicecomb_input_mat)
                policy_selectDice.train()
            else:
                # 상대는 고정 정책으로 선택 (NN 업데이트 X)
                # call_policy(type=1) 은 (action, action_tuple) 반환
                dice_ret = call_policy(dicecomb_input_mat, type=1, policy=fixed_policy_num)
                action = dice_ret[0] if isinstance(dice_ret, tuple) else dice_ret
                action = int(action) if not isinstance(action, int) else action

            if action != 9:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, action_matrix[action])
            else:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, ())

            done_flag_dice = 0 if result == -1 else 1

            # 에이전트 턴일 때만 replay 저장
            if is_agent_turn:
                replay_memory_dice.add(dicecomb_input_mat, action, get_rewards_dice, updated_state_dice, done_flag_dice)
                agent_rewards_dice += get_rewards_dice

            selectaction_input_mat = game.Learn_getState(2)

            # ---------------- Action phase ----------------
            if action != 9:
                if STOP_MASK_FLAG_THRESHOLD > 0 and len(game.flags) < STOP_MASK_FLAG_THRESHOLD:
                    action_masking = [1, 0, 0]
                else:
                    action_masking = [1, 1, 0]
            else:
                action_masking = [0, 0, 1]

            if is_agent_turn:
                policy_selectAction.eval()
                with torch.no_grad():
                    action = select_action(action_matrix, action_masking, episode, 2, policy_selectAction, selectaction_input_mat)
                policy_selectAction.train()
            else:
                # 상대 턴: boom 상태면 policy 호출하지 않고 강제로 action=2
                #  (Fixed_policy.extract_data 는 type=2 에서 is_boom 을 항상 False 로
                #   반환하므로, boom 일 때 call_policy 호출 시 action=0/1 이 돌아와 에러)
                if action_masking == [0, 0, 1]:
                    action = 2
                else:
                    act_ret = call_policy(selectaction_input_mat, type=2, policy=fixed_policy_num)
                    action = int(act_ret) if not isinstance(act_ret, int) else act_ret

            if action_masking[2] == 1 and action != 2:
                print(f"\n error \n")
                return

            game_result, get_rewards_action, updated_state_action = game.update_state_action(action, get_rewards_dice)
            done_flag_action = 0 if game_result == -1 else 1

            if is_agent_turn:
                replay_memory_action.add(selectaction_input_mat, action, get_rewards_action, updated_state_action, done_flag_action)
                agent_rewards_action += get_rewards_action

            isError, isFinished = game.check_error()
            if isError:
                return -1
            if isFinished is True and game_result == -1:
                print(f"GAME Finished BUT NOT RETURN")
                return -1

            if game_result != -1:
                # 게임 종료 — 승자 판별
                # update_state_action 에서 result=1 이면 방금 액션 건 플레이어가 승리
                # cur_player 는 이미 change_turn 으로 바뀌었을 수 있으니,
                # is_agent_turn (이번 스텝의 행위자) 기준으로 판정
                final_result = 1
                winner_is_agent = is_agent_turn
                game_step = step
                break

            # ---------------- Learning update ----------------
            if start_replay is True:
                if step % UPDATE_PER_STEP == 0:
                    profile = (updatecnt % num_epoch == 0)
                    l_d, l_a = replay_batch(replay_memory_dice, replay_memory_action, optim_dice, optim_action,
                                             target_selectDice, policy_selectDice, target_selectAction, policy_selectAction, profile)
                    episode_losses_dice.append(l_d)
                    episode_losses_action.append(l_a)
                    updated = True

        game_step += 1
        avg_step.append(game_step)
        # 에이전트만의 평균 (step으로 나눔 — step 중 절반 정도가 에이전트 턴)
        avg_rewards_dice.append(float(agent_rewards_dice) / float(game_step))
        avg_rewards_action.append(float(agent_rewards_action) / float(game_step))
        if episode_losses_dice:
            avg_loss_dice.append(np.mean(episode_losses_dice))
            avg_loss_action.append(np.mean(episode_losses_action))

        # 승률 로깅
        if final_result == 1:
            win_history.append(1 if winner_is_agent else 0)
        else:
            win_history.append(-1)   # timeout / draw
        # 최근 100 에피소드 이동 평균 (draw 제외)
        recent = [w for w in win_history[-100:] if w in (0, 1)]
        rolling_winrate.append(np.mean(recent) if recent else 0.5)

        if replay_memory_dice.__len__() >= REPLAY_LIMIT:
            start_replay = True

        if use_schedule is True and updated is True:
            schedule_dice.step()
            schedule_action.step()

        if episode % UPDATE_PER_EPISODE == 0:
            target_selectDice = copy.deepcopy(policy_selectDice)
            target_selectAction = copy.deepcopy(policy_selectAction)

    total_time = time.time() - train_start_time
    print(f"\n=== Training Complete (vs fixed_policy={fixed_policy_num}) ===")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Episodes: {EPI_LEN} | Avg ep/s: {EPI_LEN/total_time:.2f}")
    wins = sum(1 for w in win_history if w == 1)
    losses = sum(1 for w in win_history if w == 0)
    draws = sum(1 for w in win_history if w == -1)
    total_decided = wins + losses
    final_wr = (wins / total_decided) if total_decided > 0 else 0.0
    final_wr_recent = np.mean([w for w in win_history[-500:] if w in (0, 1)]) if len(win_history) >= 100 else float('nan')
    print(f"Overall winrate: {final_wr:.3f}  (W{wins} / L{losses} / D{draws})")
    print(f"Recent-500 winrate: {final_wr_recent:.3f}")
    print(f"Final avg loss - dice: {avg_loss_dice[-1]:.4f}, action: {avg_loss_action[-1]:.4f}" if avg_loss_dice else "No replay updates")

    save_model(policy_selectDice, policy_selectAction, avg_rewards_dice, avg_rewards_action, avg_step,
               None, None, None, None, use_schedule, min_lr, avg_loss_dice, avg_loss_action,
               experiment_name=experiment_name)
    # 승률 그래프도 별도 저장
    _save_winrate_plot(rolling_winrate, experiment_name)


# ======================================================================
# Plan A→B 전환: Mixed Self-Play (self-play + 고정 상대 혼합)
# ----------------------------------------------------------------------
# 목적:
#   - 순수 self-play(D 계열) → stop 편향 collapse
#   - 순수 fixed-policy(F 계열) → 한 휴리스틱에 overfit, winrate 0.4 한계
#   - 혼합: 일정 비율 fixed_policy 풀에서 샘플링한 상대와 매치, 나머지는 self-play
#     "신호 보존(fixed)" + "자기 발견(selfplay)" 양립
#
# 동작:
#   - 매 에피소드마다 random.uniform 으로 모드 결정
#       * selfplay_prob 이상 → self-play (양쪽 모두 NN, 양쪽 모두 replay 기여)
#       * selfplay_prob 미만 → vs fixed (상대 = opponent_pool 에서 랜덤 샘플,
#                                         agent_player 도 랜덤화하여 P1/P2 편향 방지)
#   - winrate 로깅은 fixed 에피소드만 집계 (self-play 승리는 무의미)
#   - 보상 / 마스킹 / 학습 루프는 기존과 동일
# ======================================================================
def learn_mixed_selfplay(selfplay_prob=0.7, opponent_pool=(1, 2, 3),
                         use_schedule=False, min_lr=1e-6, experiment_name=None):
    global device
    global ep
    print("START MAIN (mixed self-play + fixed)")
    print(f"selfplay_prob = {selfplay_prob}  (1-p = vs fixed)")
    print(f"opponent_pool = {opponent_pool}  (1=Random, 2=HoldAtThree, 3=ScoreOrBust)")
    print(f"BATCH_SIZE : {BATCH_SIZE}")
    print(f"Device is {device}")

    input_channels_dice = 7
    input_channels_action = 4
    input_width = 11
    output_dim_dice = 10
    output_dim_action = 3

    num_epoch = 200
    updatecnt = 0

    target_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice).to(device)
    policy_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice).to(device)
    target_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action).to(device)
    policy_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action).to(device)

    optim_dice, schedule_dice = optim_schedule(policy_selectDice, min_lr, lr_override=LR_DICE)
    optim_action, schedule_action = optim_schedule(policy_selectAction, min_lr)

    replay_memory_dice = Cantstop_ReplayMemory(action_size=output_dim_dice, buffer_size=REPLAY_LIMIT, batch_size=BATCH_SIZE)
    replay_memory_action = Cantstop_ReplayMemory(action_size=output_dim_action, buffer_size=REPLAY_LIMIT, batch_size=BATCH_SIZE)

    game = Cantstop()

    avg_rewards_dice = []
    avg_rewards_action = []
    avg_step = []
    avg_loss_dice = []
    avg_loss_action = []
    # 혼합 학습 전용 로그
    win_history = []           # 1=agent win, 0=lose, -1=timeout — fixed 에피소드만
    rolling_winrate = []        # fixed 에피소드 누적 (self-play 무시 → 동일 길이 유지 위해 마지막 값 복사)
    per_opp_history = {p: [] for p in opponent_pool}   # {opp: [1/0/-1, ...]}
    selfplay_count = 0
    fixed_count = 0

    start_replay = False
    updated = False
    train_start_time = time.time()
    print("START EPISODE")

    for episode in range(EPI_LEN):
        ep += 1
        game.Learn_reset()
        game_step = STEP_LIMIT - 1
        episode_losses_dice = []
        episode_losses_action = []
        rewards_dice_total = 0
        rewards_action_total = 0
        final_result = -1
        winner_is_agent = False

        # 모드 결정
        is_selfplay_ep = (random.uniform(0, 1) < selfplay_prob)
        if is_selfplay_ep:
            this_opponent = None
            agent_player = -1   # 의미 없음 (양쪽 NN)
            selfplay_count += 1
        else:
            this_opponent = random.choice(list(opponent_pool))
            agent_player = random.choice([1, 2])
            fixed_count += 1

        if episode % 50 == 0:
            elapsed = time.time() - train_start_time
            eps_per_sec = (episode / elapsed) if elapsed > 0 and episode > 0 else 0
            recent_decided = [w for w in win_history[-100:] if w in (0, 1)]
            recent_wr = np.mean(recent_decided) if recent_decided else float('nan')
            print(f"episode: {episode} | elapsed: {elapsed:.1f}s | ep/s: {eps_per_sec:.2f} | "
                  f"sp/fx: {selfplay_count}/{fixed_count} | recent fixed-winrate: {recent_wr:.3f}")

        for step in range(STEP_LIMIT):
            cur_player = game.player
            # self-play 에피소드면 항상 agent 턴, 그렇지 않으면 cur_player 와 비교
            is_agent_turn = True if is_selfplay_ep else (cur_player == agent_player)

            # ---------------- Dice phase ----------------
            action_matrix, action_masking = game.Learn_dice()
            dicecomb_input_mat = game.Learn_getState(1)

            if is_agent_turn:
                policy_selectDice.eval()
                with torch.no_grad():
                    action = select_action(action_matrix, action_masking, episode, 1, policy_selectDice, dicecomb_input_mat)
                policy_selectDice.train()
            else:
                dice_ret = call_policy(dicecomb_input_mat, type=1, policy=this_opponent)
                action = dice_ret[0] if isinstance(dice_ret, tuple) else dice_ret
                action = int(action) if not isinstance(action, int) else action

            if action != 9:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, action_matrix[action])
            else:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, ())

            done_flag_dice = 0 if result == -1 else 1

            if is_agent_turn:
                replay_memory_dice.add(dicecomb_input_mat, action, get_rewards_dice, updated_state_dice, done_flag_dice)
                rewards_dice_total += get_rewards_dice

            selectaction_input_mat = game.Learn_getState(2)

            # ---------------- Action phase ----------------
            if action != 9:
                if STOP_MASK_FLAG_THRESHOLD > 0 and len(game.flags) < STOP_MASK_FLAG_THRESHOLD:
                    action_masking = [1, 0, 0]
                else:
                    action_masking = [1, 1, 0]
            else:
                action_masking = [0, 0, 1]

            if is_agent_turn:
                policy_selectAction.eval()
                with torch.no_grad():
                    action = select_action(action_matrix, action_masking, episode, 2, policy_selectAction, selectaction_input_mat)
                policy_selectAction.train()
            else:
                # 상대 턴: boom 시 강제 action=2 (Fixed_policy.extract_data type=2 의 is_boom 항상 False 버그 우회)
                if action_masking == [0, 0, 1]:
                    action = 2
                else:
                    act_ret = call_policy(selectaction_input_mat, type=2, policy=this_opponent)
                    action = int(act_ret) if not isinstance(act_ret, int) else act_ret

            if action_masking[2] == 1 and action != 2:
                print(f"\n error \n")
                return

            game_result, get_rewards_action, updated_state_action = game.update_state_action(action, get_rewards_dice)
            done_flag_action = 0 if game_result == -1 else 1

            if is_agent_turn:
                replay_memory_action.add(selectaction_input_mat, action, get_rewards_action, updated_state_action, done_flag_action)
                rewards_action_total += get_rewards_action

            isError, isFinished = game.check_error()
            if isError:
                return -1
            if isFinished is True and game_result == -1:
                print(f"GAME Finished BUT NOT RETURN")
                return -1

            if game_result != -1:
                final_result = 1
                # self-play 에피소드는 winner_is_agent 의미없음(둘 다 agent)
                winner_is_agent = is_agent_turn if not is_selfplay_ep else True
                game_step = step
                break

            # ---------------- Learning update ----------------
            if start_replay is True:
                if step % UPDATE_PER_STEP == 0:
                    profile = (updatecnt % num_epoch == 0)
                    l_d, l_a = replay_batch(replay_memory_dice, replay_memory_action, optim_dice, optim_action,
                                             target_selectDice, policy_selectDice, target_selectAction, policy_selectAction, profile)
                    episode_losses_dice.append(l_d)
                    episode_losses_action.append(l_a)
                    updated = True

        game_step += 1
        avg_step.append(game_step)
        avg_rewards_dice.append(float(rewards_dice_total) / float(game_step))
        avg_rewards_action.append(float(rewards_action_total) / float(game_step))
        if episode_losses_dice:
            avg_loss_dice.append(np.mean(episode_losses_dice))
            avg_loss_action.append(np.mean(episode_losses_action))

        # 승률은 fixed 에피소드만 기록
        if not is_selfplay_ep:
            if final_result == 1:
                w = 1 if winner_is_agent else 0
            else:
                w = -1
            win_history.append(w)
            per_opp_history[this_opponent].append(w)
            recent = [x for x in win_history[-100:] if x in (0, 1)]
            rolling_winrate.append(np.mean(recent) if recent else 0.5)
        else:
            # self-play 에피소드: 그래프 길이 동기화 위해 직전 값 복사
            rolling_winrate.append(rolling_winrate[-1] if rolling_winrate else 0.5)

        if replay_memory_dice.__len__() >= REPLAY_LIMIT:
            start_replay = True

        if use_schedule is True and updated is True:
            schedule_dice.step()
            schedule_action.step()

        if episode % UPDATE_PER_EPISODE == 0:
            target_selectDice = copy.deepcopy(policy_selectDice)
            target_selectAction = copy.deepcopy(policy_selectAction)

    total_time = time.time() - train_start_time
    print(f"\n=== Mixed Training Complete (selfplay_prob={selfplay_prob}) ===")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Episodes: {EPI_LEN} (selfplay {selfplay_count}, fixed {fixed_count}) | Avg ep/s: {EPI_LEN/total_time:.2f}")
    decided = [w for w in win_history if w in (0, 1)]
    final_wr = np.mean(decided) if decided else 0.0
    print(f"Overall fixed winrate: {final_wr:.3f}  (over {len(decided)} decided of {len(win_history)} fixed-eps)")
    for opp, hist in per_opp_history.items():
        d = [w for w in hist if w in (0, 1)]
        if d:
            opp_name = {1: 'Random', 2: 'HoldAtThree', 3: 'ScoreOrBust'}.get(opp, f'P{opp}')
            print(f"  vs {opp_name:12s}: {np.mean(d):.3f}  (n={len(d)})")
    if avg_loss_dice:
        print(f"Final avg loss - dice: {avg_loss_dice[-1]:.4f}, action: {avg_loss_action[-1]:.4f}")

    save_model(policy_selectDice, policy_selectAction, avg_rewards_dice, avg_rewards_action, avg_step,
               None, None, None, None, use_schedule, min_lr, avg_loss_dice, avg_loss_action,
               experiment_name=experiment_name)
    _save_winrate_plot(rolling_winrate, experiment_name)


def _save_winrate_plot(rolling_winrate, experiment_name):
    """Plan A 전용: rolling winrate 그래프 저장."""
    if not rolling_winrate:
        return
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        safe_name = experiment_name.replace(' ', '_').replace('/', '-')
        # 폴더는 save_model 이 이미 만들어 둔 것과 합치기 어렵기 때문에
        # 간단히 최신 폴더를 찾아 저장하거나 별도로 파일만 저장.
        # 여기선 현재 디렉토리에 파일로 저장 (사용자가 원하면 후속 이동)
        path = f"winrate_{safe_name}_{current_time}.png"
    else:
        path = f"winrate_{current_time}.png"
    plt.figure()
    plt.plot(rolling_winrate)
    plt.title('Rolling Winrate (vs fixed opponent, last 100 decided)')
    plt.xlabel('Episodes')
    plt.ylabel('Winrate')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.savefig(path, format='png')
    plt.clf()
    print(f"Winrate plot saved: {path}")


#리플레이 메모리에서 BATCH_SIZE만큼 추출 후 target update
def replay(replay_memory_dice, replay_memory_action, optim_dice, optim_action, target_d, policy_d, target_a, policy_a):
    state_dims, actions, rewards, next_state_dims, done = replay_memory_dice.sample()
    state_dims_a, actions_a, rewards_a, next_state_dims_a, done_a = replay_memory_action.sample()
    for i in range(BATCH_SIZE):
        
        single_state_dim = state_dims[i] # (7,11)
        single_state_dim = single_state_dim.unsqueeze(0) #(1,7,11)
        single_action = actions[i]#.item()
        single_reward = rewards[i].item()
        single_next_state_dim = next_state_dims[i]
        single_next_state_dim = single_next_state_dim.unsqueeze(0)
        single_done = done[i].item()
        update_target(optim_dice, policy_d, target_d, single_state_dim, single_action, single_reward, single_next_state_dim, single_done,dice_or_action=1)

        single_state_dim_a = state_dims_a[i] # (4, 11)
        single_action_a = actions_a[i]#.item()
        single_reward_a = rewards_a[i].item()
        single_next_state_dim_a = next_state_dims_a[i]
        single_state_dim_a = single_state_dim_a.unsqueeze(0) #(1,4,11)
        single_next_state_dim_a = single_next_state_dim_a.unsqueeze(0)
        single_done_a = done_a[i].item()
        update_target(optim_action, policy_a, target_a, single_state_dim_a, single_action_a, single_reward_a, single_next_state_dim_a, single_done_a, dice_or_action=2)



       
'''
action_matrix : 가능한 모든 행동
action_masking : 그 행동 중 가능한 행동의 index는 1, 그 외에는 0
cur_episode : 현재 몇 번째 에피소드인지
dice_or_action : 주사위 조합 선택인지 ( == 1 ), 추가행동에 대한 선택인지 ( == 2 )
state : 현 state matrix
'''


#모델 저장
def save_model(policy_selectDice, policy_selectAction, avg_rewards_dice, avg_rewards_action, avg_step, avg_rewards_dice_p2 = None, avg_rewards_action_p2 = None, Winner = None, PolicyInfo = None, use_schedule = False, min_lr = -1, avg_loss_dice = None, avg_loss_action = None, experiment_name = None):
    if policy_selectDice is not None:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if experiment_name:
            # 실험 이름을 폴더명 prefix로 — 여러 실험 결과 구분 용도
            safe_name = experiment_name.replace(' ', '_').replace('/', '-')
            dir_name = f"model_{safe_name}_{current_time}"
        else:
            dir_name = f"model_{current_time}"
        os.makedirs(dir_name, exist_ok=True)
        model_path = os.path.join(dir_name, "model_dice.pth")
        torch.save(policy_selectDice.state_dict(), model_path)
        model_path = os.path.join(dir_name, "model_action.pth")
        torch.save(policy_selectAction.state_dict(), model_path)
        from Cantstop_Gamefile import Cantstop as _cs
        params_info = (
            f"EXPERIMENT : {experiment_name if experiment_name else '(unnamed)'}\n"
            f"BATCH_SIZE : {BATCH_SIZE}\n"
            f"GAMMA = {GAMMA}\nGAMMA_DICE = {GAMMA_DICE}\n"
            f"EPS_START = {EPS_START}\nEPS_END = {EPS_END}\n"
            f"EPS_END_D = {EPS_END_D}\nEPS_END_A = {EPS_END_A}\n"
            f"EPS_DECAY = {EPS_DECAY}\nTAU = {TAU}\n"
            f"LR = {LR}\nLR_DICE = {LR_DICE}\n"
            f"REPLAY_LIMIT = {REPLAY_LIMIT}\n"
            f"UPDATE_PER_EPISODE = {UPDATE_PER_EPISODE}\n"
            f"EPI_LEN = {EPI_LEN}\nSTEP_LIMIT = {STEP_LIMIT}\n"
            f"UPDATE_PER_STEP = {UPDATE_PER_STEP}, use_schedule = {use_schedule}\n"
            f"minlr = {min_lr}\n"
            f"STOP_MASK_FLAG_THRESHOLD = {STOP_MASK_FLAG_THRESHOLD}\n"
            f"DICE_BOOM_PENALTY = {_cs.DICE_BOOM_PENALTY}\n"
            f"FLAG_EXISTING_BONUS = {_cs.FLAG_EXISTING_BONUS}\n"
            f"FLAG_NEW_PENALTY = {_cs.FLAG_NEW_PENALTY}\n"
            f"CONTINUE_SAFETY_BONUS_0 = {_cs.CONTINUE_SAFETY_BONUS_0}\n"
            f"CONTINUE_SAFETY_BONUS_1 = {_cs.CONTINUE_SAFETY_BONUS_1}\n"
            f"CONTINUE_SAFETY_BONUS_2 = {_cs.CONTINUE_SAFETY_BONUS_2}\n"
        )
        params_file_path = os.path.join(dir_name, 'params.txt')

        plt.plot(avg_rewards_dice)
        plt.title('Average Rewards Dice_Function')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.savefig(os.path.join(dir_name, 'avg_rewards_dice.png'), format='png')
        plt.clf()

        plt.plot(avg_rewards_action)
        plt.title('Average Rewards action_function')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.savefig(os.path.join(dir_name, 'avg_rewards_action.png'), format='png')
        plt.clf()

        plt.plot(avg_step)
        plt.title('Average steps')
        plt.xlabel('Episodes')
        plt.ylabel('Average step')
        plt.savefig(os.path.join(dir_name, 'avg_step.png'), format='png')
        plt.clf()

        if avg_loss_dice and avg_loss_action:
            plt.plot(avg_loss_dice, label='Dice Loss')
            plt.plot(avg_loss_action, label='Action Loss')
            plt.title('Training Loss')
            plt.xlabel('Episodes (after replay start)')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.savefig(os.path.join(dir_name, 'training_loss.png'), format='png')
            plt.clf()

    else:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = f"compare policy_{PolicyInfo[0]}_vs_{PolicyInfo[1]}_time_{current_time}"
        os.makedirs(dir_name, exist_ok=True)

        plt.plot(avg_rewards_dice, label = "Policy : " + PolicyInfo[0], color = 'blue')
        plt.plot(avg_rewards_dice_p2, label = "Policy : " + PolicyInfo[1], color = 'red', linestyle = '--')
        plt.title('Average Rewards Dice_Function')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.savefig(os.path.join(dir_name, 'avg_rewards_dice.png'), format='png')
        plt.clf()

        plt.plot(avg_rewards_action, label = "Policy : " + PolicyInfo[0], color = 'blue')
        plt.plot(avg_rewards_action_p2, label = "Policy : " + PolicyInfo[1], color = 'red')
        plt.title('Average Rewards action_function')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.savefig(os.path.join(dir_name, 'avg_rewards_action.png'), format='png')
        plt.clf()

        plt.plot(avg_step)
        plt.title('Average steps')
        plt.xlabel('Episodes')
        plt.ylabel('Average step')
        plt.savefig(os.path.join(dir_name, 'avg_step.png'), format='png')
        plt.clf()

        result_counter = Counter(Winner)
        match_p1 = [result_counter[1], result_counter[0], result_counter[2]]
        match_p2 = [result_counter[2], result_counter[0], result_counter[1]]
        groups = 3
        index = np.arange(groups)
        bar_width = 0.4

        fig, ax = plt.subplots()
        bar1 = ax.bar(index - bar_width/2, match_p1, bar_width, label = PolicyInfo[0])
        bar2 = ax.bar(index + bar_width/2, match_p2, bar_width, label = PolicyInfo[1])

        ax.set_xlabel('Match Result')
        ax.set_ylabel('Counts')
        ax.set_title('Match Results Comparison between Two Players')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(['Wins', 'Draws', 'Losses'])
        ax.legend()
        
        plt.savefig(os.path.join(dir_name, 'MatchResult.png'), format = 'png')
        

        section_size = 100

        avg_dice_sections = calculate_section_averages(avg_rewards_dice, section_size)
        avg_dice_p2_sections = calculate_section_averages(avg_rewards_dice_p2, section_size)
        avg_action_sections = calculate_section_averages(avg_rewards_action, section_size)
        avg_action_p2_sections = calculate_section_averages(avg_rewards_action_p2, section_size)
    
        #sections = range(len(avg_dice_sections))  # 구간 번호
        min_length = min(len(avg_dice_sections), len(avg_dice_p2_sections), len(avg_action_sections), len(avg_action_p2_sections))
        sections = range(min_length)  # 구간 번호 조정
        bar_width = 0.35  # 막대 너비

        fig, ax = plt.subplots(2, 1, figsize=(12, 10))  # Dice와 Action에 대한 서브플롯

        # Dice Function 막대그래프
        ax[0].bar(sections, avg_dice_sections[:min_length], bar_width, label='Policy : ' + PolicyInfo[0], color='blue')
        ax[0].bar([p + bar_width for p in sections], avg_dice_p2_sections[:min_length], bar_width, label='Policy : ' + PolicyInfo[1], color='red')
        ax[0].set_title('Average Rewards per Section for Dice Function')
        ax[0].set_xlabel('Section')
        ax[0].set_ylabel('Average Reward')
        ax[0].set_xticks([p + bar_width / 2 for p in sections])
        ax[0].set_xticklabels([f'{i*section_size}-{min((i+1)*section_size-1, len(avg_rewards_dice)-1)}' for i in sections])
        ax[0].legend()

        # Action Function 막대그래프
        ax[1].bar(sections, avg_action_sections[:min_length], bar_width, label='Policy : ' + PolicyInfo[0], color='blue')
        ax[1].bar([p + bar_width for p in sections], avg_action_p2_sections[:min_length], bar_width, label='Policy : ' + PolicyInfo[1], color='red')
        ax[1].set_title('Average Rewards per Section for Action Function')
        ax[1].set_xlabel('Section')
        ax[1].set_ylabel('Average Reward')
        ax[1].set_xticks([p + bar_width / 2 for p in sections])
        ax[1].set_xticklabels([f'{i*section_size}-{min((i+1)*section_size-1, len(avg_rewards_action)-1)}' for i in sections])
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(dir_name, 'section_comparison.png'), format='png')
        plt.clf()
    if policy_selectDice is not None:
        with open(params_file_path, 'w') as file:
            file.write(params_info)


    print(f"모델과 파라미터 정보가 {dir_name} 폴더에 저장되었습니다.")

def calculate_section_averages(rewards, section_size):
    return [np.mean(rewards[i:min(i+section_size, len(rewards))]) for i in range(0, len(rewards), section_size)]

#정책에 따른 행동 선택/ E-greedy 사용
def select_action(action_matrix, action_masking, cur_episode, dice_or_action, policy, state):

    if dice_or_action == 1:
        if action_masking[9] == 1:#어떤 행동도 할 수 없는 경우 - 주사위 선택 
            return 9
    else:
        if action_masking[2] == 1:
            return 2
        
    #행동이 선택 가능할 때    
    global device
    sample = random.random()

    if dice_or_action == 1:
        eps_threshold = EPS_END_D + (EPS_START - EPS_END_D) * \
            math.exp(-1. * cur_episode / EPS_DECAY)
    else:
        eps_threshold = EPS_END_A + (EPS_START - EPS_END_A) * \
            math.exp(-1. * cur_episode / EPS_DECAY)
        
    policy = policy.to(device)
    if dice_or_action == 1:
        state = state.view(1, 7, 11)
    else:
        state = state.view(1, 4, 11)
    state_to_cuda =  state.clone().detach().to(device)


    if dice_or_action == 1:
        action_result = []#가능한 action만을 확인
        for i in range(len(action_matrix)):#action_masking을 통해 불가능한 행동은 0으로 처리
            action_result.append(action_matrix[i] if action_masking[i] else 0)
        with torch.no_grad():
            dice_action = policy(state_to_cuda)#state에서 각 행동에 대한 q값을 반환
            valid_actions = []

        for i in range(len(action_matrix)):
            if action_masking[i]:
                valid_actions.append(dice_action[0][i])
            else:
                valid_actions.append(torch.tensor(float('-inf'), device=device))

        valid_actions_tensor = torch.stack(valid_actions)

    if sample > eps_threshold:#Greedy
        if dice_or_action == 1:#주사위 조합 선택
            # 가능한 행동 중 최대 Q값을 가진 행동 찾기
            action = valid_actions_tensor.argmax().item()
        else:#행동 선택
            with torch.no_grad():
                q_values = policy(state_to_cuda).squeeze()
                for i in range(len(action_masking)):
                    if action_masking[i] == 0:
                        q_values[i] = float('-inf')
                action = q_values.argmax().item()
        return torch.tensor(action)
    else:#무작위 행동
        if dice_or_action == 1:#주사위 조합 선택
            indices_of_ones = [i for i, value in enumerate(valid_actions_tensor) if value != float('-inf')]
            action = random.choice(indices_of_ones)
        else:#행동 선택
            valid_indices = [i for i in range(len(action_masking)) if action_masking[i] == 1]
            action = random.choice(valid_indices)

        return torch.tensor(action)
    
#DQN 갱신 수식을 통한 값 업데이트
def update_target(optimizer, policy, target, state, action, reward, next_state, dones = -1, is_replay = 0, dice_or_action = 1):
    global device

    if dones != -1:
        dones = -2
    if dice_or_action == 1:
        state_reshaped = state.view(1, 7, 11).to(device)
        next_state = next_state.view(1, 7, 11).to(device)
    else:
        state_reshaped = state.view(1, 4, 11).to(device)
        next_state = next_state.view(1, 4, 11).to(device)

    if not isinstance(action, int):
        action = action.unsqueeze(-1)
    
    q_expected = policy(state_reshaped).gather(1, action)

    q_target_next = target(next_state).detach().max(1)[0].unsqueeze(1)
    q_target = reward + GAMMA * q_target_next * (2+dones)#다음 상태에서 게임이 종료되면 0, 아니면 1을 곱해주어야함\
    
    try:
        loss = F.mse_loss(q_expected, q_target)
    except UserWarning as e:
        global ep
        print(f"policy value : {np.shape(policy(state)[0][action])}")
        print(f"target value : {target(state)}")
        print(f"policy shape : {np.shape(policy(state))}")
        print(f"target shape : {np.shape(target(state))}")
        print(f"target_next shape : {np.shape(target(next_state))}")
        print(f"is_replay : {is_replay}, episode : {ep}")
        print(f"q_expected : {q_expected}, q_target_next : {q_target_next} q_target : {q_target}")
        print(f"q_expected_shape : {np.shape(q_expected)}, q_target_next_shape : {np.shape(q_target_next)} q_target_shape : {np.shape(q_target)}")

        sys.exit(1)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return q_target
def replay_batch(replay_memory_dice, replay_memory_action, optim_dice, optim_action, target_d, policy_d, target_a, policy_a, profile=False):
    # 리플레이 메모리에서 배치 샘플링
    state_dims, actions, rewards, next_state_dims, done = replay_memory_dice.sample()
    state_dims_a, actions_a, rewards_a, next_state_dims_a, done_a = replay_memory_action.sample()

    # 배치 전체를 텐서로 변환할 필요 없음, 이미 텐서임
    state_dims = state_dims.to(device)
    next_state_dims = next_state_dims.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    done = done.to(device)
    
    state_dims_a = state_dims_a.to(device)
    next_state_dims_a = next_state_dims_a.to(device)
    actions_a = actions_a.to(device)
    rewards_a = rewards_a.to(device)
    done_a = done_a.to(device)

    _, loss_dice = update_target_batch_DDQN(optim_dice, policy_d, target_d, state_dims, actions, rewards, next_state_dims, done, dice_or_action=1)
    _, loss_action = update_target_batch_DDQN(optim_action, policy_a, target_a, state_dims_a, actions_a, rewards_a, next_state_dims_a, done_a, dice_or_action=2)
    return loss_dice, loss_action
    '''
    # 업데이트 함수 호출
    if profile:
        try:
            with profiler.profile(record_shapes=True) as prof:
                update_target_batch(optim_dice, policy_d, target_d, state_dims, actions, rewards, next_state_dims, done, dice_or_action=1)
                update_target_batch(optim_action, policy_a, target_a, state_dims_a, actions_a, rewards_a, next_state_dims_a, done_a, dice_or_action=2)
            print(prof.key_averages().table(sort_by="cpu_time_total"))
        except Exception as e:
            print(f"Profiler failed: {e}")
    else:
        update_target_batch(optim_dice, policy_d, target_d, state_dims, actions, rewards, next_state_dims, done, dice_or_action=1)
        update_target_batch(optim_action, policy_a, target_a, state_dims_a, actions_a, rewards_a, next_state_dims_a, done_a, dice_or_action=2)
    '''
def update_target_batch(optimizer, policy, target, states, actions, rewards, next_states, dones, dice_or_action):
    global device

    if dice_or_action == 1:
        states = states.view(-1, 7, 11).to(device)
        next_states = next_states.view(-1, 7, 11).to(device)
    else:
        states = states.view(-1, 4, 11).to(device)
        next_states = next_states.view(-1, 4, 11).to(device)

    actions = actions.view(-1, 1).to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)

    q_expected = policy(states).gather(1, actions)
    q_target_next = target(next_states).detach().max(1)[0].unsqueeze(1)
    q_target = rewards + GAMMA * q_target_next * (1 - dones) 

    loss = torch.nn.functional.mse_loss(q_expected, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return q_target

def update_target_batch_DDQN(optimizer, policy, target, states, actions, rewards, next_states, dones, dice_or_action):
    global device

    if dice_or_action == 1:
        states = states.view(-1, 7, 11).to(device)
        next_states = next_states.view(-1, 7, 11).to(device)
        gamma = GAMMA_DICE
    else:
        states = states.view(-1, 4, 11).to(device)
        next_states = next_states.view(-1, 4, 11).to(device)
        gamma = GAMMA

    actions = actions.view(-1, 1).to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)

    q_expected = policy(states).gather(1, actions)

    next_action = policy(next_states).detach().max(1)[1].unsqueeze(1)
    q_target_next = target(next_states).gather(1, next_action).detach()

    # 🅳-철회: Minimax 부호 플립 제거 (2026-04-24)
    #  - 시도했으나 Can't Stop의 보상 구조가 순수 제로섬이 아니라
    #    (safety bonus / boom penalty / flag bonus 등은 개별 rollout 신호)
    #    부호 플립이 과잉 penalization으로 작동 → "항상 continue" 반대 편향 발생
    #  - 표준 DDQN target 으로 원복
    q_target = rewards + gamma * q_target_next * (1 - dones)

    loss = torch.nn.functional.mse_loss(q_expected, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return q_target, loss.item()


# ======================================================================
# Periodic eval: self-play 학습 중 fixed_policy 풀에 매치 → collapse 즉시 감지
# ----------------------------------------------------------------------
# self-play 는 rolling winrate 자체 측정 불가 (양쪽 다 학습 정책).
# 정해진 episode 마다 상대 풀 (Random/HoldAtThree/ScoreOrBust) 에 N 게임 매치 후
# winrate 를 snapshot 으로 기록. 학습 그래프 옆에 기록해서 mid-training collapse 조기발견.
# 평가 시 epsilon=0 (greedy), 학습 변경 X.
# ======================================================================
def _quick_eval(policy_dice, policy_action, opponent_num, n_games=100, max_step=150):
    """greedy agent vs fixed_policy 빠른 평가. Return: (wins, losses, draws)."""
    eval_game = Cantstop()
    wins = losses = draws = 0
    policy_dice.eval()
    policy_action.eval()
    for g in range(n_games):
        eval_game.Learn_reset()
        # 매번 agent_player 토글 (P1/P2 편향 제거)
        agent_player = 1 if (g % 2 == 0) else 2
        result = -1
        winner_is_agent = False
        for step in range(max_step):
            cur_player = eval_game.player
            is_agent = (cur_player == agent_player)
            # ---- Dice phase ----
            action_matrix, action_masking = eval_game.Learn_dice()
            dice_state = eval_game.Learn_getState(1)
            if is_agent:
                with torch.no_grad():
                    q = policy_dice(dice_state.unsqueeze(0).to(device)).squeeze()
                for idx, m in enumerate(action_masking):
                    if m == 0:
                        q[idx] = float('-inf')
                action = int(torch.argmax(q).item())
            else:
                dice_ret = call_policy(dice_state, type=1, policy=opponent_num)
                action = dice_ret[0] if isinstance(dice_ret, tuple) else dice_ret
                action = int(action) if not isinstance(action, int) else action
            if action != 9:
                _, get_rd, _ = eval_game.update_state_dice(action, action_matrix[action])
            else:
                _, get_rd, _ = eval_game.update_state_dice(action, ())
            # ---- Action phase ----
            if action != 9:
                if STOP_MASK_FLAG_THRESHOLD > 0 and len(eval_game.flags) < STOP_MASK_FLAG_THRESHOLD:
                    am = [1, 0, 0]
                else:
                    am = [1, 1, 0]
            else:
                am = [0, 0, 1]
            act_state = eval_game.Learn_getState(2)
            if is_agent:
                with torch.no_grad():
                    qa = policy_action(act_state.unsqueeze(0).to(device)).squeeze()
                for idx, m in enumerate(am):
                    if m == 0:
                        qa[idx] = float('-inf')
                act = int(torch.argmax(qa).item())
            else:
                if am == [0, 0, 1]:
                    act = 2
                else:
                    ar = call_policy(act_state, type=2, policy=opponent_num)
                    act = int(ar) if not isinstance(ar, int) else ar
            game_result, _, _ = eval_game.update_state_action(act, get_rd)
            isErr, isFin = eval_game.check_error()
            if isErr:
                break
            if game_result != -1:
                result = 1
                winner_is_agent = is_agent
                break
        if result == 1:
            if winner_is_agent:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1
    policy_dice.train()
    policy_action.train()
    return wins, losses, draws


def _save_eval_snapshots(eval_log, experiment_name):
    """eval_log: list of (episode, {opp: winrate}). 그래프 1장 저장."""
    if not eval_log:
        return
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = (experiment_name or 'unnamed').replace(' ', '_').replace('/', '-')
    eps = [e[0] for e in eval_log]
    opps = list(eval_log[0][1].keys())
    opp_names = {1: 'Random', 2: 'HoldAtThree', 3: 'ScoreOrBust'}
    for opp in opps:
        wrs = [e[1][opp] for e in eval_log]
        plt.plot(eps, wrs, marker='o', label=opp_names.get(opp, f'P{opp}'))
    plt.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Episodes')
    plt.ylabel('Winrate (greedy vs fixed)')
    plt.title('Periodic Eval Snapshots')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f"eval_{safe_name}_{current_time}.png", format='png')
    plt.clf()


def play_test(policy_dice, policy_action): #플레이하며 q값 등 확인

    game = Cantstop()
    game.play_init()
    game_finished = False
    isBoom = False
    while not game_finished:
        game.print_state()
        print(f"\nPrint_State\n")
        state_dice, isBoom = game.play_game_turn()
        print(f"\nPlay_Turn\n")
        if isBoom:
            print(f"Boom")
            continue
        state_dice = state_dice.unsqueeze(0)
        fixed_action, fixed_action_tuple = call_policy(state = state_dice, type = 1, policy = 2)
        q_dice = policy_dice(state_dice)
        print(f"\nCurrent Q Value in Dice : {q_dice}")
        print(f"Fixed policy's action : {fixed_action_tuple}")
        game.print_state()
        print(f"\nPrint_State_dice\n")
        state_action = game.select_dice()
        
        print("\nAction_Dice Selected \n")
        game.print_state()
        print(f"\nPrint_State_action\n")
        state_action = state_action.unsqueeze(0)
        q_action = policy_action(state_action)
        fixed_action = call_policy(state = state_action, type = 2, policy = 2)
        print(f"\nCurrent Q Value in Dice : {q_action}")
        print(f"Fixed Policy's Action : {fixed_action}. //// 0 means play more. 1 means stop.")
        game_finished = game.select_action()

        print("\nAction Selected\n")

    return 1

def update_param(batchsize = 128,
                 gamma = 0.9,
                 gamma_dice = 0.6,
                epsstart = 0.9,
                epsend = 0.1 ,
                epsend_a = None,   # action 전용 EPS_END (None이면 epsend 사용)
                epsdecay = 1400,
                tau = 0.005,
                lr = 1e-4,
                lr_dice = None,
                updateperep = 10,
                relim = 25000,
                eplen = 2000,
                stlim = 150,
                updateperstep = 5
                ):
    global BATCH_SIZE
    global GAMMA
    global GAMMA_DICE
    global EPS_START
    global EPS_END
    global EPS_DECAY
    global TAU
    global REPLAY_LIMIT
    global LR
    global LR_DICE
    global UPDATE_PER_EPISODE
    global EPI_LEN
    global STEP_LIMIT
    global EPS_END_D
    global EPS_END_A

    BATCH_SIZE = batchsize
    GAMMA = gamma
    GAMMA_DICE = gamma_dice
    EPS_START = epsstart
    EPS_END = epsend
    EPS_DECAY = epsdecay
    TAU = tau
    LR = lr
    LR_DICE = lr_dice
    UPDATE_PER_EPISODE = updateperep
    REPLAY_LIMIT = relim
    EPI_LEN = eplen
    STEP_LIMIT = stlim

    EPS_END_D = epsend
    # action 전용 epsilon 하한: 'always-stop' 고착 방지용으로 따로 설정 가능
    EPS_END_A = epsend if epsend_a is None else epsend_a

    UPDATE_PER_STEP = updateperstep

def call_policy(state,type,policy, DQNModel = None):
    if policy == 1:
        if type == 1:
            action_matrix, action_masking = fp.extract_data(state, type, policy)
            action, action_tuple = fp.policy_all_random_dice(action_matrix, action_masking)
            return action, action_tuple
        else:
            is_boom = fp.extract_data(state,type,policy)
            action = fp.policy_all_random_action(is_boom)
            return action
    elif policy == 2:
        if type == 1:
            action_matrix, action_masking, flags = fp.extract_data(state, type, policy)
            action, action_tuple = fp.policy_Hold_At_Three_Flags_dice(action_matrix, action_masking, flags) 
            return action, action_tuple
        else:
            #flags, action_masking = fp.extract_data(state, type, policy)
            #action = fp.policy_Hold_At_Three_Flags_action(flags, action_masking)
            flags, action_masking, player_progress, turn_player_progress = fp.extract_data(state, type, policy)
            action = fp.policy_Hold_At_Three_Flags_action_v2(flags, action_masking, player_progress, turn_player_progress)
            return action
    elif policy == 3:
        if type == 1:
            action_matrix, action_masking, flags, turn_player_progress = fp.extract_data(state, type, policy)
            action, action_tuple = fp.policy_Score_Or_Bust_dice(action_matrix, action_masking, flags, turn_player_progress)
            return action, action_tuple
        else:
            is_boom, player_progress, turn_player_progress, flags = fp.extract_data(state, type, policy)
            action = fp.policy_Score_Or_Bust_action(is_boom, player_progress, turn_player_progress, flags)
            return action
    else:
        if DQNModel is None:
            print(f"unavaliable policy number. current policy is : {policy}")
        else:
            if type == 1:
                epsilon = 0.1 #epsilon_greedy 사용 시
                action_matrix, action_masking = fp.extract_data(state, type, policy)
            
                state = state.unsqueeze(0)
                q_value = DQNModel(state).squeeze()
                #print(f"actionmat : {action_matrix}\nactionmask : {action_masking}")
                #print(f"qval is : {q_value}")
                for idx, masking in enumerate(action_masking):
                    if masking == 0:
                        q_value[idx] = float('-inf')

                if random.uniform(0,1) < epsilon:
                    valid_actions = [idx for idx, val in enumerate(action_masking) if val != 0]
                    best_action = random.choice(valid_actions)
                else:
                    best_action = torch.argmax(q_value)
                #print(f"qval after masking : {q_value}")
                #print(f"best action is : {best_action}")
                #print(f"action is {action_matrix[best_action] if best_action != 9 else 'boom'}")
                
                if best_action == 9:
                    return best_action, None
                return best_action, action_matrix[best_action]
            else:
                is_boom = fp.extract_data(state, type, policy)
                if is_boom == True:
                    return 2
                else:
                    epsilon = 0.1 #dice만 greedy로 한다거나의 선택지도 있으니 따로 선언

                    #print("\n\n    ACTION PHASE      \n\n")
                    state = state.unsqueeze(0)
                    q_value = DQNModel(state).squeeze()
                    #print(f"qval is : {q_value}")
                    q_value[2] = float('-inf')
                    #print(f"qval after masking : {q_value}")
                    if random.uniform(1,0) < epsilon:
                        valid_actions = [0,1]
                        best_action = random.choice(valid_actions)
                    else:
                        best_action = torch.argmax(q_value)
                    #print(f"best action is : {best_action}")                    
                    return best_action
                
def compare_policy_stats(policy_p1, policy_p2, policy_dice = None, policy_action = None):#policy number
    global ep

    game = Cantstop()
    avg_rewards_dice_p1 = []
    avg_rewards_action_p1 = []
    avg_rewards_dice_p2 = []
    avg_rewards_action_p2 = []
    avg_step = []
    winner = []
    type_dice = 1
    type_action = 2
    cur_player = -1
    policy = [policy_p1, policy_p2]
    policyText = ["All_Random", "Hold_At_Three", "Score_or_Bust", "DQN Model"]
    boomedplayer = -1
    lastaction_d = -1
    lastaction_a = -1
    continous_action_cnt = 0
    for episode in range(EPI_LEN):
        ep += 1
        game.Learn_reset()
        game_step = STEP_LIMIT - 1
        cur_player = game.player
        cur_policy = policy[cur_player-1]
        rewards_dice_p1 = 0
        rewards_dice_p2 = 0
        rewards_action_p1 = 0
        rewards_action_p2 = 0
        if episode % 10 == 0:
            print(f"current_episode : {episode}")
        for step in range(STEP_LIMIT):
            cur_player = game.player
            cur_policy = policy[cur_player-1]
            
            action_matrix, action_masking = game.Learn_dice()

            dicecomb_input_mat = game.Learn_getState(1)
            action, action_tuple = call_policy(dicecomb_input_mat, type_dice, cur_policy, policy_dice)

            if action != 9:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, action_matrix[action])
            else:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, None)

            if cur_player == 1:
                rewards_dice_p1 += get_rewards_dice
            else:
                rewards_dice_p2 += get_rewards_dice


            selectaction_input_mat = game.Learn_getState(2)
            
            if action != 9:
                action_masking = [1, 1, 0]
                action = call_policy(selectaction_input_mat, type_action, cur_policy, policy_action)
            else:
                action_masking = [0, 0, 1]
                action = 2
                boomedplayer = cur_player
            lastaction_a = action

            if lastaction_a == 0:
                continous_action_cnt += 1
            else:
                continous_action_cnt = 0


            game_result, get_rewards_action, updated_state_action = game.update_state_action(action, get_rewards_dice)

            if cur_player == 1:
                rewards_action_p1 += get_rewards_action
            else:
                rewards_action_p2 += get_rewards_action

            
            isError, isFinished = game.check_error()

            if isError:
                return -1
            
            if isFinished is True and game_result == -1:
                print(f"GAME Finished BUT NOT RETURN")
                return -1
            
            if game_result != -1:
                game_step = step
                winner.append(cur_player) #Save Match's Winner
                avg_step.append(game_step)
                break

        if game_result == -1:    
            game_step += 1
            avg_step.append(game_step)
            winner.append(0)#Draw
        avg_rewards_dice_p1.append(float(rewards_dice_p1) / float(game_step))
        avg_rewards_action_p1.append(float(rewards_action_p1) / float(game_step))   
        avg_rewards_dice_p2.append(float(rewards_dice_p2) / float(game_step))
        avg_rewards_action_p2.append(float(rewards_action_p2) / float(game_step))   

    policy_info = [policyText[policy_p1-1], policyText[policy_p2-1]]
    save_model(None, None, avg_rewards_dice_p1, avg_rewards_action_p1, avg_step, avg_rewards_dice_p2, avg_rewards_action_p2, winner, policy_info)

    return

def optim_schedule(policyNet, min_lr = 1e-06, lr_override=None):
    initial_lr = lr_override if lr_override is not None else LR
    ep_len = EPI_LEN
    decay_rate = (min_lr / initial_lr) ** (1 / ep_len)
    optimizer = optim.Adam(policyNet.parameters(), lr=initial_lr)
    scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    return optimizer, scheduler

if __name__ == '__main__':

    #learn_or_play = input("\n\nLearn : 1 // Play : 2 :: ")
    #  2 -> play_test (대국 플레이)
    #  3 -> self-play 배치 학습 (D1~D7 / E1 계열)
    #  4 -> Plan A: 고정 상대 학습 (F1~F6)
    #  5 -> Plan A→B: Mixed self-play + 고정 상대 풀 (M1~M3)
    #  6 -> M1~M3 평가 배치 (per-opponent winrate)
    #  7 -> Plan C: Self-play 재진입 + periodic eval (S1~S8)
    #  8 -> Plan D: REPLAY_LIMIT sweep (R1~R3) — collapse 진단
    #  기타 -> DQN vs Fixed policy 대결 평가
    learn_or_play = 8
    learn_or_play = int(learn_or_play)
    if learn_or_play == 3:
        # =============================================================
        # 버그 수정 후 첫 실험 세트 (D1~D7, overnight batch)
        # =============================================================
        # 전제 변경: change_turn 에서 state 추출 타이밍 수정 (perspective 정상화).
        # 이전 실험들은 stop next_state가 왜곡돼 Q(stop) 과대평가.
        # 이제부터 safety bonus / GAMMA / mask 효과가 **정상적으로** 측정됨.
        #
        # 실험 축:
        #  - Safety bonus 스케일: high(1.5) / mid(0.5) / low(0.3) / flag=2 강화
        #  - GAMMA: 0.5 (기준) / 0.3 / 0.7
        #  - STOP 마스킹: ON(=2) / OFF(=0)
        #  - EPS_END_A: 0.15 / 0.05
        # 공통 고정:
        #  - FLAG_EXISTING_BONUS=0.15, FLAG_NEW_PENALTY=0.3
        #  - lr=5e-5, lr_dice=2e-4, batch=512, epi=8000, step_lim=150
        #  - tau=0.01, gamma_dice=0.5
        # =============================================================
        import Cantstop_Gamefile as game_module

        def _set_rewards(flag_ex, flag_new, sb0, sb1, sb2):
            game_module.Cantstop.FLAG_EXISTING_BONUS = flag_ex
            game_module.Cantstop.FLAG_NEW_PENALTY = flag_new
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_0 = sb0
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_1 = sb1
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_2 = sb2

        # 공통 update_param 기본값 (실험별로 덮어쓰기)
        common = dict(
            relim=50000, lr=5e-5, lr_dice=2e-4, batchsize=512,
            gamma=0.5, gamma_dice=0.5, epsdecay=1500, eplen=8000,
            stlim=150, epsstart=0.99, epsend=0.01, epsend_a=0.15,
            updateperep=10, updateperstep=3, tau=0.01,
        )

        experiments = [
            # E1: D2 설정 + Minimax 부호 플립 (bootstrap 턴 전환 시 -1 부호)
            #  - D2가 7개 중 유일하게 roll 선택이 관찰돼 "bootstrap이 덜 지배적일수록 정상"임을 시사
            #  - 여기에 minimax 플립을 적용하면 Q(stop) 과대평가의 근원이 제거됨
            {
                'name': 'E1_minimax_D2',
                'rewards': (0.15, 0.3, 0.5, 0.3, 0.1),
                'mask_threshold': 2,
                'params': dict(common),
            },
        ]

        total = len(experiments)
        for idx, exp in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] Running experiment: {exp['name']}")
            print(f"  rewards (fe,fn,sb0,sb1,sb2) = {exp['rewards']}")
            print(f"  mask_threshold = {exp['mask_threshold']}")
            print(f"  params override: gamma={exp['params']['gamma']}, "
                  f"epsend_a={exp['params']['epsend_a']}")
            print(f"{'='*60}\n")

            # 리워드/환경 파라미터 세팅
            _set_rewards(*exp['rewards'])
            # 모듈 레벨 변수 업데이트 (__main__ 블록은 모듈 레벨이라 직접 할당)
            STOP_MASK_FLAG_THRESHOLD = exp['mask_threshold']

            # 학습 파라미터
            update_param(**exp['params'])

            # 학습 실행 (내부에서 save_model이 experiment_name prefix로 폴더 저장)
            learn(use_schedule=False, min_lr=-1, experiment_name=exp['name'])

        print(f"\n{'='*60}")
        print(f"All {total} experiments completed. Check model_D*_* folders.")
        print(f"{'='*60}\n")

    elif learn_or_play == 4:
        # =============================================================
        # Plan A: 고정 룰베이스 상대 학습 (self-play collapse 회피)
        # =============================================================
        # 배경: D1~D7 self-play 학습 결과 모두 "stop 편향 → Fixed_policy 에 참패".
        # 두 쪽이 같은 약한 정책으로 수렴해 승패 신호 희석(self-play collapse) 의심.
        # 여기선 상대를 고정 룰베이스(Hold_At_Three / Score_or_Bust)로 두고
        # 에이전트만 학습. "일찍 stop = 진다" 라는 신호가 직접적으로 전달되기를 기대.
        # =============================================================
        import Cantstop_Gamefile as game_module

        def _set_rewards(flag_ex, flag_new, sb0, sb1, sb2):
            game_module.Cantstop.FLAG_EXISTING_BONUS = flag_ex
            game_module.Cantstop.FLAG_NEW_PENALTY = flag_new
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_0 = sb0
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_1 = sb1
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_2 = sb2

        # Plan A: self-play 보다 에피소드 수 축소 (상대 고정이라 더 빨리 수렴 기대)
        common = dict(
            relim=50000, lr=5e-5, lr_dice=2e-4, batchsize=512,
            gamma=0.5, gamma_dice=0.5, epsdecay=1500, eplen=4000,
            stlim=150, epsstart=0.99, epsend=0.01, epsend_a=0.15,
            updateperep=10, updateperstep=3, tau=0.01,
        )

        experiments = [
            # --------------------------------------------------------------
            # [축 1] 상대 비교: Hold_At_Three vs Score_or_Bust
            # --------------------------------------------------------------
            # F1: 공격적 상대 (3flag 까지 무조건 굴림)
            #     → "굴려야 이긴다" 신호는 약함. 상대가 bust 나면 얻어걸리는 식
            {
                'name': 'F1_vs_HoldAtThree',
                'opponent': 2,
                'rewards': (0.15, 0.3, 1.5, 1.5, 1.5),  # D1 base (strong guidance)
                'mask_threshold': 2,
                'params': dict(common),
            },
            # F2: 균형잡힌 휴리스틱 상대. 가장 "학습 기대값 높은" 상대.
            #     → 1순위 baseline
            {
                'name': 'F2_vs_ScoreOrBust',
                'opponent': 3,
                'rewards': (0.15, 0.3, 1.5, 1.5, 1.5),
                'mask_threshold': 2,
                'params': dict(common),
            },

            # --------------------------------------------------------------
            # [축 2] Reward shaping 강도: D2 중간(0.5~0.1) vs 제거(0)
            # --------------------------------------------------------------
            # F3: 중간 shaping — self-play 에선 랜덤도 못 이긴 D2 설정.
            #     고정 상대 환경에서도 이러면 shaping 자체가 부족한 것.
            {
                'name': 'F3_vs_ScoreOrBust_lowSafety',
                'opponent': 3,
                'rewards': (0.15, 0.3, 0.5, 0.3, 0.1),
                'mask_threshold': 2,
                'params': dict(common),
            },
            # F4: Shaping 완전 제거 — sparse-ish reward
            #     → stop 시 conquer/win 만 양의 보상. 나머진 boom penalty 정도.
            #     승패 신호가 강한 고정 상대 환경에서 sparse 가 가능한지 테스트
            {
                'name': 'F4_vs_ScoreOrBust_noShaping',
                'opponent': 3,
                'rewards': (0.0, 0.0, 0.0, 0.0, 0.0),
                'mask_threshold': 2,
                'params': dict(common),
            },

            # --------------------------------------------------------------
            # [축 3] 도메인 지식 의존도: stop masking OFF
            # --------------------------------------------------------------
            # F5: Mask 제거 — flag<2 에서도 stop 허용.
            #     고정 상대 환경에선 "너무 일찍 stop=진다"가 명확하므로
            #     agent 가 스스로 적정 stop 기준을 학습하는지 확인
            {
                'name': 'F5_vs_ScoreOrBust_noMask',
                'opponent': 3,
                'rewards': (0.15, 0.3, 1.5, 1.5, 1.5),
                'mask_threshold': 0,
                'params': dict(common),
            },

            # --------------------------------------------------------------
            # [축 4] 시간 지평: gamma 상향
            # --------------------------------------------------------------
            # F6: 더 긴 지평(gamma=0.7) — 상대의 누적 위협을 더 잘 반영하는지
            #     self-play 에선 bootstrap 편향으로 역효과였으나,
            #     고정 상대는 예측 가능한 rollout 이라 longer horizon 이 유리할 수도
            {
                'name': 'F6_vs_ScoreOrBust_gamma07',
                'opponent': 3,
                'rewards': (0.15, 0.3, 1.5, 1.5, 1.5),
                'mask_threshold': 2,
                'params': dict(common, gamma=0.7),
            },
        ]

        total = len(experiments)
        for idx, exp in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] Plan A experiment: {exp['name']}")
            print(f"  opponent policy = {exp['opponent']}")
            print(f"  rewards (fe,fn,sb0,sb1,sb2) = {exp['rewards']}")
            print(f"  mask_threshold = {exp['mask_threshold']}")
            print(f"  params: gamma={exp['params']['gamma']}, eplen={exp['params']['eplen']}")
            print(f"{'='*60}\n")

            _set_rewards(*exp['rewards'])
            STOP_MASK_FLAG_THRESHOLD = exp['mask_threshold']
            update_param(**exp['params'])

            learn_vs_fixed(
                fixed_policy_num=exp['opponent'],
                use_schedule=False,
                min_lr=-1,
                experiment_name=exp['name'],
                agent_player=1,
            )

        print(f"\n{'='*60}")
        print(f"Plan A: {total} experiments completed. Check model_F*_* folders + winrate_*.png")
        print(f"{'='*60}\n")

    elif learn_or_play == 5:
        # =============================================================
        # Plan A→B: Mixed Self-Play (self-play + 고정 상대 풀)
        # =============================================================
        # F1~F6 결과: 4000 episode 동안 winrate 0.2~0.4. 상승 추세 분명하나
        # 단일 휴리스틱 한 명에 overfit 가능성. 또한 self-play 신호도 잃음.
        # 여기선 매 에피소드마다 일정 확률로 self-play vs 고정 상대 선택,
        # 상대도 [Random, HoldAtThree, ScoreOrBust] 풀에서 랜덤 샘플 → 일반화 + 신호 보존.
        #
        # 실험 축: selfplay 비중 6:4 / 7:3 / 8:2 (selfplay_prob = 0.6 / 0.7 / 0.8)
        # EPI_LEN = 8000 (F 계열 4000보다 2배, self-play 분량 확보)
        # =============================================================
        import Cantstop_Gamefile as game_module

        def _set_rewards(flag_ex, flag_new, sb0, sb1, sb2):
            game_module.Cantstop.FLAG_EXISTING_BONUS = flag_ex
            game_module.Cantstop.FLAG_NEW_PENALTY = flag_new
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_0 = sb0
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_1 = sb1
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_2 = sb2

        common = dict(
            relim=50000, lr=5e-5, lr_dice=2e-4, batchsize=512,
            gamma=0.5, gamma_dice=0.5, epsdecay=1500, eplen=8000,
            stlim=150, epsstart=0.99, epsend=0.01, epsend_a=0.15,
            updateperep=10, updateperstep=3, tau=0.01,
        )

        # 보상 / 마스킹은 F1 이 가장 우수했던 설정 차용
        # (D1 base shaping + flag mask=2)
        common_rewards = (0.15, 0.3, 1.5, 1.5, 1.5)
        common_mask = 2

        experiments = [
            # M1: self-play 60% : fixed 40% — 신호 가장 강함, self-play 비중 가장 작음
            {
                'name': 'M1_mix_60sp_40fx',
                'selfplay_prob': 0.6,
                'opponent_pool': (1, 2, 3),
                'rewards': common_rewards,
                'mask_threshold': common_mask,
                'params': dict(common),
            },
            # M2: self-play 70% : fixed 30% — 균형
            {
                'name': 'M2_mix_70sp_30fx',
                'selfplay_prob': 0.7,
                'opponent_pool': (1, 2, 3),
                'rewards': common_rewards,
                'mask_threshold': common_mask,
                'params': dict(common),
            },
            # M3: self-play 80% : fixed 20% — self-play 우위, 신호는 소량으로만 주입
            {
                'name': 'M3_mix_80sp_20fx',
                'selfplay_prob': 0.8,
                'opponent_pool': (1, 2, 3),
                'rewards': common_rewards,
                'mask_threshold': common_mask,
                'params': dict(common),
            },
        ]

        total = len(experiments)
        for idx, exp in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] Mixed experiment: {exp['name']}")
            print(f"  selfplay_prob = {exp['selfplay_prob']}  (1-p = vs fixed)")
            print(f"  opponent_pool = {exp['opponent_pool']}")
            print(f"  rewards (fe,fn,sb0,sb1,sb2) = {exp['rewards']}")
            print(f"  mask_threshold = {exp['mask_threshold']}")
            print(f"  params: gamma={exp['params']['gamma']}, eplen={exp['params']['eplen']}")
            print(f"{'='*60}\n")

            _set_rewards(*exp['rewards'])
            STOP_MASK_FLAG_THRESHOLD = exp['mask_threshold']
            update_param(**exp['params'])

            learn_mixed_selfplay(
                selfplay_prob=exp['selfplay_prob'],
                opponent_pool=exp['opponent_pool'],
                use_schedule=False,
                min_lr=-1,
                experiment_name=exp['name'],
            )

        print(f"\n{'='*60}")
        print(f"Plan A→B Mixed: {total} experiments completed. Check model_M*_* folders + winrate_*.png")
        print(f"{'='*60}\n")

    elif learn_or_play == 6:
        # =============================================================
        # M1~M3 평가 배치: 각 모델 × {Random, HoldAtThree, ScoreOrBust}
        # =============================================================
        # 학습 중 rolling winrate 는 3종 상대 혼합 평균이라 진짜 강도 불명.
        # 여기선 모델별로 각 상대를 따로 측정해 per-opponent winrate 로깅.
        # =============================================================
        update_param(eplen=2000, stlim=150)

        input_channels_dice = 7
        input_channels_action = 4
        input_width = 11
        output_dim_dice = 10
        output_dim_action = 3

        # 모델 폴더 경로 (CWD 가 어디든 절대 경로로 지정)
        model_root = r"C:\RL_Sybilla\RL_CantStop"
        model_folders = [
            ("M1_60sp_40fx", os.path.join(model_root, "model_M1_mix_60sp_40fx_20260425-140602")),
            ("M2_70sp_30fx", os.path.join(model_root, "model_M2_mix_70sp_30fx_20260425-154424")),
            ("M3_80sp_20fx", os.path.join(model_root, "model_M3_mix_80sp_20fx_20260425-172222")),
        ]
        opponents = [(1, "Random"), (2, "HoldAtThree"), (3, "ScoreOrBust")]

        for model_label, folder in model_folders:
            print(f"\n{'#'*60}")
            print(f"## EVAL MODEL: {model_label}")
            print(f"## folder: {folder}")
            print(f"{'#'*60}")

            policy_dice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice)
            policy_action = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action)

            dice_state_dict = torch.load(os.path.join(folder, "model_dice.pth"))
            action_state_dict = torch.load(os.path.join(folder, "model_action.pth"))
            policy_dice.load_state_dict(dice_state_dict)
            policy_action.load_state_dict(action_state_dict)

            for opp_num, opp_name in opponents:
                print(f"\n--- {model_label} (DQN=4) vs {opp_name}({opp_num}) ---")
                compare_policy_stats(4, opp_num, policy_dice, policy_action)

        print(f"\n{'#'*60}")
        print(f"## M1~M3 evaluation complete. Check console winner stats above.")
        print(f"{'#'*60}\n")

    elif learn_or_play == 7:
        # =============================================================
        # Plan C: Self-play 재진입 — reward shaping 균형 + horizon 변주
        # =============================================================
        # F/M 결과: fixed_policy 학습은 reward 가 정의한 정책으로 수렴 (상대 무관).
        # → "최적 정책의 source" 는 결국 self-play. 다시 self-play 로 가되,
        # D 시리즈가 collapse 한 원인 (sb 값 / gamma 가 win signal 압도) 을
        # 정확히 사이 영역(D ↔ F) 에서 sweep.
        # 모든 실험에 periodic eval (1500ep 마다 fixed pool 80게임) 으로 mid-training collapse 즉시 감지.
        # =============================================================
        import Cantstop_Gamefile as game_module

        def _set_rewards(flag_ex, flag_new, sb0, sb1, sb2):
            game_module.Cantstop.FLAG_EXISTING_BONUS = flag_ex
            game_module.Cantstop.FLAG_NEW_PENALTY = flag_new
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_0 = sb0
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_1 = sb1
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_2 = sb2

        common = dict(
            relim=50000, lr=5e-5, lr_dice=2e-4, batchsize=512,
            gamma=0.5, gamma_dice=0.5, epsdecay=1500, eplen=8000,
            stlim=150, epsstart=0.99, epsend=0.01, epsend_a=0.15,
            updateperep=10, updateperstep=3, tau=0.01,
        )

        experiments = [
            # --- [축 1] sb 균일 vs escalating, gamma=0.7 ---
            # S1: F의 1.5 → 0.7 균일 약화 (가장 보수적 변경)
            {
                'name': 'S1_sb07uni_g07',
                'rewards': (0.15, 0.3, 0.7, 0.7, 0.7),
                'mask_threshold': 2,
                'params': dict(common, gamma=0.7),
                'epsend_a': 0.15,
            },
            # S2: escalating (flag 늘수록 sb ↑) — "초반 안전, 후반 공격" 학습 유도
            {
                'name': 'S2_sbEsc_g07',
                'rewards': (0.15, 0.3, 0.5, 0.7, 1.0),
                'mask_threshold': 2,
                'params': dict(common, gamma=0.7),
                'epsend_a': 0.15,
            },

            # --- [축 2] gamma ↑ → win signal 우위 강화 ---
            # S3: 균일 sb + gamma 0.85
            {
                'name': 'S3_sb07uni_g085',
                'rewards': (0.15, 0.3, 0.7, 0.7, 0.7),
                'mask_threshold': 2,
                'params': dict(common, gamma=0.85),
                'epsend_a': 0.15,
            },
            # S4: escalating + gamma 0.85
            {
                'name': 'S4_sbEsc_g085',
                'rewards': (0.15, 0.3, 0.5, 0.7, 1.0),
                'mask_threshold': 2,
                'params': dict(common, gamma=0.85),
                'epsend_a': 0.15,
            },

            # --- [축 3] 더 sparse 한 영역 (D4 noShaping 실패 회피하면서 sparse 에 근접) ---
            # S5: 매우 약한 shaping + gamma 0.9
            {
                'name': 'S5_sbWeak_g09',
                'rewards': (0.05, 0.1, 0.3, 0.5, 0.7),
                'mask_threshold': 2,
                'params': dict(common, gamma=0.9),
                'epsend_a': 0.15,
            },

            # --- [축 4] 탐험 강화 (self-play 동일정책 collapse 회피) ---
            # S6: S3 + eps_end_a 상향 (15% → 30%)
            {
                'name': 'S6_sb07uni_g085_explore',
                'rewards': (0.15, 0.3, 0.7, 0.7, 0.7),
                'mask_threshold': 2,
                'params': dict(common, gamma=0.85),
                'epsend_a': 0.30,
            },

            # --- [축 5] 마스크 제거 → agent 가 stop 시점 스스로 학습 ---
            # S7: S4 + mask off
            {
                'name': 'S7_sbEsc_g085_noMask',
                'rewards': (0.15, 0.3, 0.5, 0.7, 1.0),
                'mask_threshold': 0,
                'params': dict(common, gamma=0.85),
                'epsend_a': 0.15,
            },

            # --- [축 6] 모든 자유 변수 결합 ---
            # S8: escalating + gamma 0.9 + 탐험↑ + mask off (가장 자유로운 학습)
            {
                'name': 'S8_sbEsc_g09_explore_noMask',
                'rewards': (0.15, 0.3, 0.5, 0.7, 1.0),
                'mask_threshold': 0,
                'params': dict(common, gamma=0.9),
                'epsend_a': 0.30,
            },
        ]

        total = len(experiments)
        for idx, exp in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] Plan C self-play: {exp['name']}")
            print(f"  rewards (fe,fn,sb0,sb1,sb2) = {exp['rewards']}")
            print(f"  mask_threshold = {exp['mask_threshold']}")
            print(f"  gamma = {exp['params']['gamma']}, eps_end_a = {exp['epsend_a']}")
            print(f"{'='*60}\n")

            _set_rewards(*exp['rewards'])
            STOP_MASK_FLAG_THRESHOLD = exp['mask_threshold']
            params = dict(exp['params'])
            params['epsend_a'] = exp['epsend_a']
            update_param(**params)

            learn(
                use_schedule=False,
                min_lr=-1,
                experiment_name=exp['name'],
                eval_every=1500,           # 1500ep 마다 평가 (8000ep 동안 5회)
                eval_n_games=80,           # 상대당 80게임 (총 240, ~1~2분 오버헤드)
                eval_opponents=(1, 2, 3),
            )

        print(f"\n{'='*60}")
        print(f"Plan C: {total} self-play experiments completed.")
        print(f"  - model_S*_*  folders   (가중치 + 학습 그래프)")
        print(f"  - eval_S*_*.png         (mid-training fixed-pool winrate snapshots)")
        print(f"{'='*60}\n")

    elif learn_or_play == 8:
        # =============================================================
        # Plan D: REPLAY_LIMIT sweep (S 시리즈 collapse 진단)
        # =============================================================
        # S1~S7 결과: 모두 ep1500 peak, 이후 단조 감소 (보상/γ/mask 무관).
        # 가설: REPLAY=50000 buffer 가 ep~1300 에 가득 참 = peak 시점과 일치.
        #       초반 무작위 (eps=0.99) 데이터가 buffer 의 대부분 → 정책 끌어내림.
        # 검증: REPLAY_LIMIT 만 변주 (5000/10000/20000), 나머지는 S2 와 동일 (escalating sb, γ=0.7).
        # 작은 buffer → 초반 데이터 빨리 밀려나감 → collapse 발생 시점 / 정도 변화 봐야 함.
        # eval_every 도 1500 → 1000 으로 좁힘 (peak 더 일찍 나올 가능성 대비).
        # =============================================================
        import Cantstop_Gamefile as game_module

        def _set_rewards(flag_ex, flag_new, sb0, sb1, sb2):
            game_module.Cantstop.FLAG_EXISTING_BONUS = flag_ex
            game_module.Cantstop.FLAG_NEW_PENALTY = flag_new
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_0 = sb0
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_1 = sb1
            game_module.Cantstop.CONTINUE_SAFETY_BONUS_2 = sb2

        # S2 베이스 (escalating sb, γ=0.7) — S 시리즈 중 H@3 / SoB 가장 덜 떨어짐
        common_base = dict(
            lr=5e-5, lr_dice=2e-4, batchsize=512,
            gamma=0.7, gamma_dice=0.5, epsdecay=1500, eplen=8000,
            stlim=150, epsstart=0.99, epsend=0.01, epsend_a=0.15,
            updateperep=10, updateperstep=3, tau=0.01,
        )
        common_rewards = (0.15, 0.3, 0.5, 0.7, 1.0)
        common_mask = 2

        experiments = [
            # R1: 매우 작은 buffer — 최근 ~80 ep transition 만 유지
            #     초반 무작위 데이터가 가장 빨리 사라짐. 단점: 학습 안정성 저하 가능
            {
                'name': 'R1_replay_5k',
                'replay_limit': 5000,
                'rewards': common_rewards,
                'mask_threshold': common_mask,
                'params': dict(common_base, relim=5000),
            },
            # R2: 중간 — 최근 ~150 ep
            {
                'name': 'R2_replay_10k',
                'replay_limit': 10000,
                'rewards': common_rewards,
                'mask_threshold': common_mask,
                'params': dict(common_base, relim=10000),
            },
            # R3: 작아진 baseline — 최근 ~330 ep
            #     원본 50000 보다 훨씬 작지만 학습 안정성은 더 잘 유지될 것
            {
                'name': 'R3_replay_20k',
                'replay_limit': 20000,
                'rewards': common_rewards,
                'mask_threshold': common_mask,
                'params': dict(common_base, relim=20000),
            },
        ]

        total = len(experiments)
        for idx, exp in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] Plan D replay sweep: {exp['name']}")
            print(f"  REPLAY_LIMIT = {exp['replay_limit']}")
            print(f"  rewards (fe,fn,sb0,sb1,sb2) = {exp['rewards']}")
            print(f"  mask_threshold = {exp['mask_threshold']}")
            print(f"  gamma = {exp['params']['gamma']}, eps_end_a = {exp['params']['epsend_a']}")
            print(f"{'='*60}\n")

            _set_rewards(*exp['rewards'])
            STOP_MASK_FLAG_THRESHOLD = exp['mask_threshold']
            update_param(**exp['params'])

            learn(
                use_schedule=False,
                min_lr=-1,
                experiment_name=exp['name'],
                eval_every=1000,           # peak 가 더 일찍 나올 수 있어 1500→1000
                eval_n_games=80,
                eval_opponents=(1, 2, 3),
            )

        print(f"\n{'='*60}")
        print(f"Plan D: {total} replay-limit experiments completed.")
        print(f"  - model_R*_*  folders + eval_R*_*.png")
        print(f"  핵심 비교: peak 위치가 buffer fill 시점에 비례해 이동하는지 확인")
        print(f"{'='*60}\n")

    elif learn_or_play == 2:
        input_channels_dice = 7
        input_channels_action = 4
        input_width = 11
        output_dim_dice = 10
        output_dim_action = 3

        policy_dice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice)
        policy_action = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action)

        dice_state_dict = torch.load(os.path.join("model_dice.pth"))
        action_state_dict = torch.load(os.path.join("model_action.pth"))

        policy_dice.load_state_dict(dice_state_dict)
        policy_action.load_state_dict(action_state_dict)
        play_test(policy_dice, policy_action)
    elif learn_or_play == 3:
        # === 전체 정책 비교: 고정 baseline 3회 + DQN 비교 3회 = 총 6회 ===
        update_param(eplen=2000, stlim=150)

        # # Part 1: 고정 정책 baseline (3C2 = 3회)
        # print("=== Part 1: Fixed Policy Baselines ===")
        # print("RUN 1/6: All Random(1) vs Hold_At_Three(2)")
        # compare_policy_stats(1, 2)
        # print("RUN 2/6: All Random(1) vs Score_Or_Bust(3)")
        # compare_policy_stats(1, 3)
        # print("RUN 3/6: Hold_At_Three(2) vs Score_Or_Bust(3)")
        # compare_policy_stats(2, 3)

        # Part 2: DQN vs 고정 정책 (3회)
        print("\n=== Part 2: DQN Model vs Fixed Policies ===")
        input_channels_dice = 7
        input_channels_action = 4
        input_width = 11
        output_dim_dice = 10
        output_dim_action = 3

        policy_dice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice)
        policy_action = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action)

        dice_state_dict = torch.load(os.path.join("model_dice.pth"))
        action_state_dict = torch.load(os.path.join("model_action.pth"))
        policy_dice.load_state_dict(dice_state_dict)
        policy_action.load_state_dict(action_state_dict)

        print("RUN 4/6: All Random(1) vs DQN(4)")
        compare_policy_stats(4, 1, policy_dice, policy_action)
        print("RUN 5/6: Hold_At_Three(2) vs DQN(4)")
        compare_policy_stats(4, 2, policy_dice, policy_action)
        print("RUN 6/6: Score_Or_Bust(3) vs DQN(4)")
        compare_policy_stats(4, 3, policy_dice, policy_action)
        print("=== All Comparisons Complete ===")
    else:
        '''
        policy with all random : 1
        hold at three flags : 2
        score or bust : 3
        DQNModel : 4
        '''
        input_channels_dice = 7
        input_channels_action = 4
        input_width = 11
        output_dim_dice = 10
        output_dim_action = 3

        policy_dice = Cantstop_DQN.SelectDiceCombDQN(input_channels_dice, input_width, output_dim_dice)
        policy_action = Cantstop_DQN.SelectAdditionalActionDQN(input_channels_action, input_width, output_dim_action)

        dice_state_dict = torch.load(os.path.join("model_dice.pth"))
        action_state_dict = torch.load(os.path.join("model_action.pth"))

        compare_policy_stats(4, 1, policy_dice, policy_action)
        compare_policy_stats(4, 2, policy_dice, policy_action)
        compare_policy_stats(4, 3, policy_dice, policy_action)







'''
conquered_flag에 대한 처리를 turn_player_progress의 기준으로 두고, 점수 계산은 저장 시에만 하는 것으로 수정해야겠는데
- Learn_check_Flags에서 current_conquered_flag로 우선 수정해봄

conquered_flag의 기준(아마 상관 없는것같긴 함)
- gamefile 내에서는 플레이어별로 처리를 하지만 dqn 학습용으로는 내 기준이 무조건 1이다
턴이 전환되지 않는 버그가 존재?


change_turn에서 dice에 관련된 처리도 있어야하던가?
update_state_dice, update_state_action과 함께 검토해봐야
'''