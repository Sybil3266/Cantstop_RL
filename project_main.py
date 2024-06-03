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
ep = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#random.seed(6)
#학습 진행
def learn(use_schedule = False, min_lr = 1e-6):
    global device
    global ep
    print("START MAIN") 
    print(f"BATCH_SIZE : {BATCH_SIZE}")

    print(f"Device is {device}")
    #input_dim_dice = 77
    #input_dim_action = 44
    input_channels = 1
    input_height_dice = 7
    input_height_action = 4
    input_width = 11
    output_dim_dice = 10
    output_dim_action = 3

    num_epoch = 200
    updatecnt = 0
    #target_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_dim_dice, output_dim_dice).to(device)
    #policy_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_dim_dice, output_dim_dice).to(device)
    #target_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_dim_action,output_dim_action).to(device)
    #policy_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_dim_action,output_dim_action).to(device)
    # 모델 초기화 부분 수정
    target_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_channels, input_height_dice, input_width, output_dim_dice).to(device)
    policy_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_channels, input_height_dice, input_width, output_dim_dice).to(device)
    target_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_channels, input_height_action, input_width, output_dim_action).to(device)
    policy_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_channels, input_height_action, input_width, output_dim_action).to(device)

    optim_dice, schedule_dice = optim_schedule(policy_selectDice, min_lr)
    optim_action, schedule_action = optim_schedule(policy_selectAction, min_lr)

    replay_memory_dice = Cantstop_ReplayMemory(action_size = output_dim_dice, buffer_size = REPLAY_LIMIT, batch_size = BATCH_SIZE)
    replay_memory_action = Cantstop_ReplayMemory(action_size = output_dim_action, buffer_size = REPLAY_LIMIT, batch_size = BATCH_SIZE)

    game = Cantstop()

    avg_rewards_dice = []
    avg_rewards_action = []
    avg_step = []
    start_replay = False
    updated = False
    print("START EPISODE")
    for episode in range(EPI_LEN):
        ep += 1
        game.Learn_reset()
        game_step = STEP_LIMIT - 1
        if episode % 10 == 0:
            print(f"current_episode : {episode}")
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

            get_rewards_dice -= 0.3
            rewards_dice += get_rewards_dice
            if result == -1:
                replay_memory_dice.add(dicecomb_input_mat, action, get_rewards_dice, updated_state_dice, -1)
            else:
                replay_memory_dice.add(dicecomb_input_mat, action, get_rewards_dice, updated_state_dice, 1)
            selectaction_input_mat = game.Learn_getState(2)

            if action != 9:
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

            game_result, get_rewards_action, updated_state_action = game.update_state_action(action)
            #get_rewards_action -= 1 #action에 대해서는 
            rewards_action += get_rewards_action
            replay_memory_action.add(selectaction_input_mat, action, get_rewards_action, updated_state_action, game_result)

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
                    replay_batch(replay_memory_dice, replay_memory_action, optim_dice, optim_action, target_selectDice, policy_selectDice, target_selectAction, policy_selectAction, profile)
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

        if replay_memory_dice.__len__() >= REPLAY_LIMIT:
            start_replay = True

        if use_schedule is True and updated is True:
            schedule_dice.step()
            schedule_action.step()

        if episode % UPDATE_PER_EPISODE == 0:
            target_selectDice = copy.deepcopy(policy_selectDice)
            target_selectAction = copy.deepcopy(policy_selectAction)

    save_model(policy_selectDice, policy_selectAction, avg_rewards_dice, avg_rewards_action, avg_step, None, None, None, None, use_schedule, min_lr)

#리플레이 메모리에서 BATCH_SIZE만큼 추출 후 target update
def replay(replay_memory_dice, replay_memory_action, optim_dice, optim_action, target_d, policy_d, target_a, policy_a):
    state_dims, actions, rewards, next_state_dims, done = replay_memory_dice.sample()
    state_dims_a, actions_a, rewards_a, next_state_dims_a, done_a = replay_memory_action.sample()
    for i in range(BATCH_SIZE):
        
        single_state_dim = state_dims[i] # (7,11)
        single_state_dim = single_state_dim.unsqueeze(0).unsqueeze(0) #(1,1,7,11)
        single_action = actions[i]#.item()
        single_reward = rewards[i].item()
        single_next_state_dim = next_state_dims[i]
        single_next_state_dim = single_next_state_dim.unsqueeze(0).unsqueeze(0)        
        single_done = done[i].item()
        update_target(optim_dice, policy_d, target_d, single_state_dim, single_action, single_reward, single_next_state_dim, single_done,dice_or_action=1)
        
        single_state_dim_a = state_dims_a[i] # (4, 11)
        single_state_dim = single_state_dim.unsqueeze(0).unsqueeze(0) #(1,1,4,11)
        single_action_a = actions_a[i]#.item()
        single_reward_a = rewards_a[i].item()
        single_next_state_dim_a = next_state_dims_a[i]
        single_state_dim_a = single_state_dim_a.unsqueeze(0).unsqueeze(0)
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
def save_model(policy_selectDice, policy_selectAction, avg_rewards_dice, avg_rewards_action, avg_step, avg_rewards_dice_p2 = None, avg_rewards_action_p2 = None, Winner = None, PolicyInfo = None, use_schedule = False, min_lr = -1):
    if policy_selectDice is not None:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = f"model_{current_time}"
        os.makedirs(dir_name, exist_ok=True)
        model_path = os.path.join(dir_name, "model_dice.pth")
        torch.save(policy_selectDice.state_dict(), model_path)
        model_path = os.path.join(dir_name, "model_action.pth")
        torch.save(policy_selectAction.state_dict(), model_path)
        params_info = f"BATCH_SIZE : {BATCH_SIZE}\n GAMMA = {GAMMA}\nEPS_START = {EPS_START}\nEPS_END = {EPS_END}\nEPS_END_D = {EPS_END_D}\nEPS_END_A = {EPS_END_A}\nEPS_DECAY = {EPS_DECAY}\nTAU = {TAU}\nLR = {LR}\nREPLAY_LIMIT = {REPLAY_LIMIT}\nUPDATE_PER_EPISODE = {UPDATE_PER_EPISODE}\nEPI_LEN = {EPI_LEN}\nSTEP_LIMIT = {STEP_LIMIT}\nUPDATE_PER_STEP = {UPDATE_PER_STEP}, use_schedule = {use_schedule}\nminlr = {min_lr}\n"
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
        state = state.view(1,1,7,11)
    else:
        state = state.view(1,1,4,11)
    state_to_cuda =  state.clone().detach().to(device) 


    if dice_or_action == 1:
        action_result = []#가능한 action만을 확인
        for i in range(len(action_matrix)):#action_masking을 통해 불가능한 행동은 0으로 처리
            action_result.append(action_matrix[i] if action_masking[i] else 0)
        with torch.no_grad():
            dice_action = policy(state_to_cuda)#state에서 각 행동에 대한 q값을 반환
            valid_actions = []

        for i in range(len(action_matrix)):
            valid_actions.append(dice_action[0][i] * action_masking[i])#불가능한 행동은 전부 q값에 0을 곱해 처리

        valid_actions_tensor = torch.tensor(valid_actions, device=device)
    
    if sample > eps_threshold:#Greedy
        if dice_or_action == 1:#주사위 조합 선택
            # 가능한 행동 중 최대 Q값을 가진 행동 찾기
            valid_action_indices = torch.nonzero(valid_actions_tensor, as_tuple=True)[0]
            if len(valid_action_indices) > 0:

                max_q_value_index = valid_actions_tensor[valid_action_indices].max(0)[1]

                action = valid_action_indices[max_q_value_index].item()

            else:
                indices_of_ones = [i for i, value in enumerate(valid_actions_tensor) if value != 0]
                action = random.choice(indices_of_ones)  # 가능한 행동이 없는 경우 일단 무작위로 처리, 탐험 하면서 값은 갱신되고, 초기에 0으로 초기화될 가능성 생각
        else:#행동 선택
            with torch.no_grad():
                q_values = policy(state_to_cuda)
                action = q_values.max(1)[1].item()  # 가장 높은 Q 값을 가지는 행동의 인덱스
        return torch.tensor(action)
    else:#무작위 행동
        if dice_or_action == 1:#주사위 조합 선택
            indices_of_ones = [i for i, value in enumerate(valid_actions_tensor) if value != 0]
            action = None
            while action == None:
                action =  random.choice(indices_of_ones)
        else:#행동 선택
            action = random.choice([0,1])

        return torch.tensor(action)
    
#DQN 갱신 수식을 통한 값 업데이트
def update_target(optimizer, policy, target, state, action, reward, next_state, dones = -1, is_replay = 0, dice_or_action = 1):
    global device

    if dones != -1:
        dones = -2
    if dice_or_action == 1:
        state_reshaped = state.view(1, 1, 7, 11).to(device)
        next_state = next_state.view(1, 1, 7, 11).to(device)
    else:
        state_reshaped = state.view(1, 1, 4, 11).to(device)
        next_state = next_state.view(1, 1, 4, 11).to(device)

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

    update_target_batch_DDQN(optim_dice, policy_d, target_d, state_dims, actions, rewards, next_state_dims, done, dice_or_action=1)
    update_target_batch_DDQN(optim_action, policy_a, target_a, state_dims_a, actions_a, rewards_a, next_state_dims_a, done_a, dice_or_action=2)
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
        states = states.view(-1, 1, 7, 11).to(device)
        next_states = next_states.view(-1, 1, 7, 11).to(device)
    else:
        states = states.view(-1, 1, 4, 11).to(device)
        next_states = next_states.view(-1, 1, 4, 11).to(device)

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
        states = states.view(-1, 1, 7, 11).to(device)
        next_states = next_states.view(-1, 1, 7, 11).to(device)
    else:
        states = states.view(-1, 1, 4, 11).to(device)
        next_states = next_states.view(-1, 1, 4, 11).to(device)

    actions = actions.view(-1, 1).to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)

    q_expected = policy(states).gather(1, actions)

    next_action = policy(next_states).detach().max(1)[1].unsqueeze(1)
    q_target_next = target(next_states).gather(1, next_action).detach()

    q_target = rewards + GAMMA * q_target_next * (1 - dones)  

    loss = torch.nn.functional.mse_loss(q_expected, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return q_target


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
                epsstart = 0.9,
                epsend = 0.1 ,
                epsdecay = 1400,
                tau = 0.005,
                lr = 1e-4,
                updateperep = 10,
                relim = 25000,
                eplen = 2000,
                stlim = 150,
                updateperstep = 5
                ):
    global BATCH_SIZE
    global GAMMA
    global EPS_START
    global EPS_END
    global EPS_DECAY
    global TAU
    global REPLAY_LIMIT
    global LR
    global UPDATE_PER_EPISODE
    global EPI_LEN
    global STEP_LIMIT
    global EPS_END_D
    global EPS_END_A

    BATCH_SIZE = batchsize
    GAMMA = gamma
    EPS_START = epsstart
    EPS_END = epsend
    EPS_DECAY = epsdecay
    TAU = tau
    LR = lr
    UPDATE_PER_EPISODE = updateperep
    REPLAY_LIMIT = relim
    EPI_LEN = eplen
    STEP_LIMIT = stlim

    EPS_END_A = epsend
    EPS_END_D = epsend

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

            get_rewards_dice -= 0.1
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


            game_result, get_rewards_action, updated_state_action = game.update_state_action(action)

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

def optim_schedule(policyNet, min_lr = 1e-06):
    initial_lr = LR
    ep_len = EPI_LEN
    decay_rate = (min_lr / initial_lr) ** (1 / ep_len)
    optimizer = optim.Adam(policyNet.parameters(), lr=initial_lr)
    scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    return optimizer, scheduler

if __name__ == '__main__':

    #learn_or_play = input("\n\nLearn : 1 // Play : 2 :: ")
    learn_or_play = 1 #1 -> learn extract model, 3 -> compare with fixed policy
    learn_or_play = int(learn_or_play)
    if learn_or_play == 1:
        epdecay = [1000]
        param_lr = [1e-04, 5e-05, 1e-05]
        use_sched = [False, True]
        minlr = [1e-06, 5e-06]

        for l in param_lr:
            for n in use_sched:
                if n:  
                    for k in minlr:
                        if l == k: 
                            continue
                        update_param(relim=20000, lr=l, batchsize=256, gamma=0.87, epsdecay=500, eplen=2000, stlim=150, epsstart=0.99, epsend=0.01)
                        learn(use_schedule=n, min_lr=k)
                else:  
                    update_param(relim=20000, lr=l, batchsize=256, gamma=0.87, epsdecay=500, eplen=4000, stlim=150, epsstart=0.99, epsend=0.01)
                    learn(use_schedule=n, min_lr=-1)

        # update_param(relim = 20000, lr = 1e-04, batchsize=256, gamma=0.87, epsdecay= 1000, eplen=4000, stlim=150, epsstart=0.9, epsend=0.01)#, relim=i)
        # learn(use_schedule=True, min_lr=1e-05)

        #update_param(batchsize=256, gamma=0.87, epsdecay=500, lr=1e-05, relim=10000, updateperep=10, eplen=2000)
        #learn()

    elif learn_or_play == 2:
        input_channels = 1
        input_height_dice = 7
        input_height_action = 4
        input_width = 11
        output_dim_dice = 10
        output_dim_action = 3

        policy_dice = Cantstop_DQN.SelectDiceCombDQN(input_channels, input_height_dice, input_width, output_dim_dice)
        policy_action = Cantstop_DQN.SelectAdditionalActionDQN(input_channels, input_height_action, input_width, output_dim_action)

        dice_state_dict = torch.load(os.path.join("model_dice.pth"))
        action_state_dict = torch.load(os.path.join("model_action.pth"))
        
        policy_dice.load_state_dict(dice_state_dict)
        policy_action.load_state_dict(action_state_dict)
        play_test(policy_dice, policy_action)
    else:
        '''
        policy with all random : 1
        hold at three flags : 2
        score or bust : 3
        DQNModel : 4
        '''
        input_channels = 1
        input_height_dice = 7
        input_height_action = 4
        input_width = 11
        output_dim_dice = 10
        output_dim_action = 3

        policy_dice = Cantstop_DQN.SelectDiceCombDQN(input_channels, input_height_dice, input_width, output_dim_dice)
        policy_action = Cantstop_DQN.SelectAdditionalActionDQN(input_channels, input_height_action, input_width, output_dim_action)

        dice_state_dict = torch.load(os.path.join("model_dice.pth"))
        action_state_dict = torch.load(os.path.join("model_action.pth"))

        compare_policy_stats(1, 4, policy_dice, policy_action)
        compare_policy_stats(2, 4, policy_dice, policy_action)
        #compare_policy_stats(3, 4, policy_dice, policy_action)







'''
conquered_flag에 대한 처리를 turn_player_progress의 기준으로 두고, 점수 계산은 저장 시에만 하는 것으로 수정해야겠는데
- Learn_check_Flags에서 current_conquered_flag로 우선 수정해봄

conquered_flag의 기준(아마 상관 없는것같긴 함)
- gamefile 내에서는 플레이어별로 처리를 하지만 dqn 학습용으로는 내 기준이 무조건 1이다
턴이 전환되지 않는 버그가 존재?


change_turn에서 dice에 관련된 처리도 있어야하던가?
update_state_dice, update_state_action과 함께 검토해봐야
'''