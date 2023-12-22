from Cantstop_Gamefile import Cantstop
import Cantstop_DQN
from Cantstop_ReplayMemory import Cantstop_ReplayMemory
import torch
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import random
import math
import copy
import numpy as np
import os
import datetime


import warnings
import sys
warnings.filterwarnings("error")

BATCH_SIZE = 64
GAMMA = 0.8 
EPS_START = 0.9
EPS_END = 0.1 
EPS_DECAY = 700
TAU = 0.005
LR = 1e-4
REPLAY_LIMIT = 100
UPDATE_PER_EPISODE = 10
EPI_LEN = 4000
STEP_LIMIT = 300
ep = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    global device
    global ep
    print("START MAIN")
    print(f"BATCH_SIZE : {BATCH_SIZE}")

    print(f"Device is {device}")
    input_dim_dice = 77
    input_dim_action = 44
    output_dim_dice = 10
    output_dim_action = 3
    target_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_dim_dice, output_dim_dice).to(device)
    policy_selectDice = Cantstop_DQN.SelectDiceCombDQN(input_dim_dice, output_dim_dice).to(device)
    target_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_dim_action,output_dim_action).to(device)
    policy_selectAction = Cantstop_DQN.SelectAdditionalActionDQN(input_dim_action,output_dim_action).to(device)
    optim_dice = optim.Adam(policy_selectDice.parameters(), lr=0.001)
    optim_action = optim.Adam(policy_selectAction.parameters(), lr=0.001)

    replay_memory_dice = Cantstop_ReplayMemory(action_size = output_dim_dice, buffer_size = 1000, batch_size = BATCH_SIZE)
    replay_memory_action = Cantstop_ReplayMemory(action_size = output_dim_action, buffer_size = 1000, batch_size = BATCH_SIZE)

    game = Cantstop()

    avg_rewards_dice = []
    avg_rewards_action = []
    avg_step = []
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
            #print(f"action_dice : {action}")
            #print(f"selected by matrix = {action_matrix[action]}")
            if action != 9:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, action_matrix[action])
            else:
                result, get_rewards_dice, updated_state_dice = game.update_state_dice(action, ())

            rewards_dice += get_rewards_dice
            #target_selectDice = 
            #print(f"state : {dicecomb_input_mat}")
            #print(f"action_index : {action} action : {action_matrix[action] if action != 9 else action}")
            #print(f"flags : {game.flags}, actionmat : {action_matrix}")
            #print(f"updated state : {updated_state_dice}")

            #print(f"single state : {dicecomb_input_mat}")
            #update_target(optim_dice, policy_selectDice, target_selectDice, dicecomb_input_mat,  action, get_rewards_dice, updated_state_dice, result)
            

            #print(f"rewards_dice : {get_rewards_dice}")
            if result == -1:
                replay_memory_dice.add(dicecomb_input_mat, action, get_rewards_dice, updated_state_dice, -1)
            else:
                replay_memory_dice.add(dicecomb_input_mat, action, get_rewards_dice, updated_state_dice, 1)
            #print("add finished")
            selectaction_input_mat = game.Learn_getState(2)

            policy_selectAction.eval()
            with torch.no_grad():
                action = select_action(action_matrix, action_masking, episode, 2, policy_selectAction, selectaction_input_mat)
            policy_selectAction.train()
            #print(f"action : {action}")
            #if action == 0:
            #    print(f"play more dice")
            #else:
            #    print(f"saved action")
            #print(f"action_select : {action}")
            #game_result, get_rewards_action = game.update_state_action(action)
            game_result, get_rewards_action, updated_state_action = game.update_state_action(action)
            rewards_action += get_rewards_action
            replay_memory_action.add(selectaction_input_mat, action, get_rewards_action, updated_state_action, game_result)
            #print(f"rewards_action : {get_rewards_action}")
            #target_selectAction = 
            #update_target(optim_action, policy_selectAction, target_selectAction, selectaction_input_mat, action, get_rewards_action, updated_state_action, game_result)

            if game_result != -1:
                game_step = step
                break
            
        game_step += 1
        avg_step.append(game_step)
        avg_rewards_dice.append(float(rewards_dice) / float(game_step))
        avg_rewards_action.append(float(rewards_action) / float(game_step))   

        if episode % UPDATE_PER_EPISODE == 0:
            target_selectDice = copy.deepcopy(policy_selectDice)
            target_selectAction = copy.deepcopy(policy_selectAction)

        if replay_memory_dice.__len__() >= REPLAY_LIMIT: #and episode % UPDATE_PER_EPISODE == 0:
            #print(f"call_replay : episode : {episode} ")
            replay(replay_memory_dice, replay_memory_action, optim_dice, optim_action, target_selectDice, policy_selectDice, target_selectAction, policy_selectAction)
    #return
    save_model(policy_selectDice, policy_selectAction, avg_rewards_dice, avg_rewards_action, avg_step)
      
def replay(replay_memory_dice, replay_memory_action, optim_dice, optim_action, target_d, policy_d, target_a, policy_a):
    state_dims, actions, rewards, next_state_dims, done = replay_memory_dice.sample()
    state_dims_a, actions_a, rewards_a, next_state_dims_a, done_a = replay_memory_action.sample()
    for i in range(BATCH_SIZE):
        single_state_dim = state_dims[i]
        single_action = actions[i].item()
        #print(f"single action : {single_action}")
        single_reward = rewards[i].item()
        single_next_state_dim = next_state_dims[i]
        single_done = done[i].item()
        update_target(optim_dice, policy_d, target_d, single_state_dim, single_action, single_reward, single_next_state_dim, single_done,)
        single_state_dim_a = state_dims_a[i]
        single_action_a = actions_a[i].item()
        single_reward_a = rewards_a[i].item()
        single_next_state_dim_a = next_state_dims_a[i]
        single_done_a = done_a[i].item()
        update_target(optim_action, policy_a, target_a, single_state_dim_a, single_action_a, single_reward_a, single_next_state_dim_a, single_done_a)



       
'''
action_matrix : 가능한 모든 행동
action_masking : 그 행동 중 가능한 행동의 index는 1, 그 외에는 0
cur_episode : 현재 몇 번째 에피소드인지
dice_or_action : 주사위 조합 선택인지 ( == 1 ), 추가행동에 대한 선택인지 ( == 2 )
state : 현 state matrix
'''



def save_model(policy_selectDice, policy_selectAction, avg_rewards_dice, avg_rewards_action, avg_step):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"model_{current_time}"
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, "model_dice.pth")
    torch.save(policy_selectDice.state_dict(), model_path)
    model_path = os.path.join(dir_name, "model_action.pth")
    torch.save(policy_selectAction.state_dict(), model_path)
    params_info = f"GAMMA = {GAMMA}\nEPS_START = {EPS_START}\nEPS_END = {EPS_END}\nEPS_DECAY = {EPS_DECAY}\nTAU = {TAU}\nLR = {LR}\nREPLAY_LIMIT = {REPLAY_LIMIT}\nUPDATE_PER_EPISODE = {UPDATE_PER_EPISODE}\nEPI_LEN = {EPI_LEN}\nSTEP_LIMIT = {STEP_LIMIT}\n"
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


    with open(params_file_path, 'w') as file:
        file.write(params_info)


    print(f"모델과 파라미터 정보가 {dir_name} 폴더에 저장되었습니다.")


def select_action(action_matrix, action_masking, cur_episode, dice_or_action, policy, state):
    global device
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * cur_episode / EPS_DECAY)
    policy = policy.to(device)
    state_to_cuda =  state.clone().detach().to(device)
    #print(f"device : {state_to_cuda.device}\nstate : {state} \nstate_to_cuda : {state_to_cuda}")
    
    #print(f"policy : {next(policy.parameters()).device}, state : {state.device}")
    action_result = []#가능한 action만을 확인
    for i in range(len(action_matrix)):#action_masking을 통해 불가능한 행동은 0으로 처리
        action_result.append(action_matrix[i] if action_masking[i] else 0)

    if dice_or_action == 1:
        with torch.no_grad():
            dice_action = policy(state_to_cuda)#state에서 각 행동에 대한 q값을 반환
            valid_actions = []
            #print(f"dice_action {dice_action}")
            #print(f"action_masking {action_masking}")

        for i in range(len(action_matrix)):
            valid_actions.append(dice_action[0][i] * action_masking[i])#불가능한 행동은 전부 q값에 0을 곱해 처리
        #print(f"valid Actions : {valid_actions}")
        valid_actions_tensor = torch.tensor(valid_actions)
    if action_masking[9] == 1:#어떤 행동도 할 수 없는 경우
        if dice_or_action == 1:
            return 9
        else:
            return 2
    
    if sample > eps_threshold:#Greedy
        if dice_or_action == 1:
            # 가능한 행동 중 최대 Q값을 가진 행동 찾기
            valid_action_indices = torch.nonzero(valid_actions_tensor, as_tuple=True)[0]
            if len(valid_action_indices) > 0:
                #print(f"valid Actions : {valid_actions_tensor}")
                #print(f"validactionindieces = {valid_action_indices}")
                max_q_value_index = valid_actions_tensor[valid_action_indices].max(0)[1]
                #print(f"maxqidx : {max_q_value_index}")
                action = valid_action_indices[max_q_value_index].item()
                #print(f"action : {action}")
            else:
                indices_of_ones = [i for i, value in enumerate(valid_actions_tensor) if value != 0]
                action = random.choice(indices_of_ones)  # 가능한 행동이 없는 경우 일단 무작위로 처리, 탐험 하면서 값은 갱신되고, 초기에 0으로 초기화될 가능성 생각
        else:
            with torch.no_grad():
                q_values = policy(state_to_cuda)
                action = q_values.max(1)[1].item()  # 가장 높은 Q 값을 가지는 행동의 인덱스
        return torch.tensor(action)
    else:#무작위 행동
        if dice_or_action == 1:
            indices_of_ones = [i for i, value in enumerate(valid_actions_tensor) if value != 0]
            action = None
            while action == None:
                action =  random.choice(indices_of_ones)
        else:
            action = random.choice([0,1])

        return torch.tensor(action)

def update_target(optimizer, policy, target, state, action, reward, next_state, dones = -1, is_replay = 0):
    global device
    #print(f"dones : {dones}. types : {dones.dtype if type(dones) is not int else type(dones)}")
    if dones != -1:
        dones = -2
    state = state.to(device)
    #action = action.to(device)
    next_state = next_state.to(device)

    #if is_replay:
    #    q_expected = policy(state)[0][action].unsqueeze(0)
    #else:
    q_expected = policy(state)[0][action].unsqueeze(0).unsqueeze(1)
    #q_expected = policy(state)[0][action].unsqueeze(1)
    q_target_next = target(next_state).detach().max(1)[0].unsqueeze(1)
    q_target = reward + GAMMA * q_target_next * (2+dones)#다음 상태에서 게임이 종료되면 0, 아니면 1을 곱해주어야함\
    
    #loss = F.mse_loss(q_expected, q_target)
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

if __name__ == '__main__':
    main()
'''
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

'''
