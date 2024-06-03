from Cantstop_Gamefile import Cantstop
import Cantstop_DQN
from Cantstop_ReplayMemory import Cantstop_ReplayMemory
import torch
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import random
import numpy as np
import math
import copy
import numpy as np
import os
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Fixed_policies for evaluate models

all random -> select all actions in random

Hold at three flags -> if flag's length became 3, save current states.
Score or Bust -> play turn until get 1 scores.


'''

'''
dice returns 9 -> there is no possible actions
action returns 0 -> throw more dice
       returns 1 -> stop and save state
       returns 2 -> there is no possible actions(dice returns 9)
'''

def extract_data(state, type, policy):
    '''
    type : 1 -> dice, 2-> action
    policy : 1 -> random, 2 -> hold at three flags, 3 -> score or bust, 4 -> DQNNet
    '''
    is_boom = False
    if type == 1:
        state = state.reshape(7,11)
    else:
        state = state.reshape(4,11)
    player_progress_current = state[0, :]  # 첫 번째 행
    player_progress_opponent = state[1, :]  # 두 번째 행
    turn_player_progress = state[2, :]  # 세 번째 행
    conquered_flag = state[3, :]  # 네 번째 행    
    if type == 1:#dice
        map_combinations = [-1] * 11
        possible_actions1 = state[4, :]
        possible_actions2 = state[5, :]
        possible_actions3 = state[6, :]
        actions = [possible_actions1, possible_actions2, possible_actions3]
        flags = []
        action_matrix = []
        possible_num = []
        action_masking = []
        for index, action in enumerate(actions):
            action_tuple = []
            for number, possible in enumerate(action):
                if possible == 1:
                    action_tuple.append(number+2)
                    possible_num.append(number+2)
                    map_combinations[number] = index
                elif possible == 2:#(7,7)같은 쌍은 앞의 행동에서 처리되기에 possible_num에서 처리하면 안됨
                    action_tuple.append(number+2)
                    action_tuple.append(number+2)
                    map_combinations[number] = index
                    #possible_num.append(number+2)
            if len(action_tuple) != 0:
                action_tuple = tuple(action_tuple)
                action_matrix.append(action_tuple)
            else:
                action_matrix.append(None)
        possible_num.sort()
        
        for num in possible_num:
            num_tuple = [num]
            num_tuple = tuple(num_tuple)
            action_matrix.append(num_tuple)

        for _ in range(9 - len(action_matrix)):
            action_matrix.append(None)

        for flg, (x, y) in enumerate(zip(turn_player_progress, player_progress_current)):
            if x > y:
                flags.append(flg+2)

        action_masking = extract_action_masking(flags=flags, turn_player_progress=turn_player_progress, player_progress_opponent=player_progress_opponent, action_matrix = action_matrix, map_combinations = map_combinations)
        if action_masking[9] == 1:
            is_boom = True
        if policy == 1:
            # checker = False
            # '''
            # mapsize = np.array([3,5,7,9,11,13,11,9,7,5,3])
            # for idx, val in enumerate(player_progress_current):
            #     if mapsize[idx] < val:
            #         print(f"PPC idx : {idx} val : {val}, maplim : {mapsize[idx]}")
            #         checker = True
            # for idx, val in enumerate(player_progress_opponent):
            #     if mapsize[idx] < val:
            #         print(f"PPO idx : {idx} val : {val}, maplim : {mapsize[idx]}")
            #         checker = True
            # for idx, val in enumerate(turn_player_progress):
            #     if mapsize[idx] < val:
            #         print(f"TPP idx : {idx} val : {val}, maplim : {mapsize[idx]}")
            #         checker = True
            # '''
            # if checker:        
            #     print(f"ppc : {player_progress_current}")
            #     print(f"ppo : {player_progress_opponent}")
            #     print(f"tpp : {turn_player_progress}")
            #     print(f"cqf : {conquered_flag}")
            #     print(f"flg : {flags}")
            #     print(f"acmat : {action_matrix}")
            #     print(f"mask : {action_masking}")

                # return -1
            return action_matrix, action_masking
        elif policy == 2:
            return action_matrix, action_masking, flags
        elif policy == 3:
            return action_matrix, action_masking, flags, turn_player_progress
        elif policy == 4:
            return action_matrix, action_masking
    else:#action
        flags = []
        for flg, (x, y) in enumerate(zip(turn_player_progress, player_progress_current)):
            if x > y:
                flags.append(flg)
        if policy == 1:
            return is_boom
        elif policy == 2:
            #return flags, is_boom
            return flags, is_boom, player_progress_current, turn_player_progress
        elif policy == 3:
            return is_boom, player_progress_current, turn_player_progress, flags
        elif policy == 4:
            return is_boom


#####################################################
def policy_all_random_dice(action_matrix, action_masking):

    valid_actions_idx = []
    for i in range(len(action_masking)):
        if action_masking[i] == 1:
            valid_actions_idx.append(i)

    selected_action = random.choice(valid_actions_idx)

    if selected_action != 9:
        return selected_action, action_matrix[selected_action]
    else:
        return selected_action, None

def policy_all_random_action(is_boom):
    if is_boom is True:
        return 2
    else:
        selected_action = random.choice([0,1])
        return selected_action

#####################################################

def policy_Hold_At_Three_Flags_dice(action_matrix, action_masking, flags):

    if action_masking[9] == 1:
        return 9, None

    valid_actions = []
    valid_actions_idx = []
    for idx, action in enumerate(action_matrix):
        if action_masking[idx] != 0:
            valid_actions.append(action)
            valid_actions_idx.append(idx)
        else:
            valid_actions.append(None)
            valid_actions_idx.append(None)

    check_camps_in_flags = []
    increase_camps_flags = []

    for action in valid_actions:
        increase_camps = 0
        if action is None:
            increase_camps += 10
        else:
            if action[0] not in flags:
                increase_camps += 1
            
            if len(action) > 1 and action[0] != action[1] and action[1] not in flags:
                increase_camps += 1
        increase_camps_flags.append(increase_camps)

    for action in valid_actions:
        camps_in_flags = 0
        if action is not None:
            if action[0] in flags:
                camps_in_flags += 1
            if len(action) > 1 and action[1] in flags:
                camps_in_flags += 1
        check_camps_in_flags.append(camps_in_flags)
    '''
    for action in valid_actions:
        camps_in_flags = 0
        if action is None:
            check_camps_in_flags.append(camps_in_flags)
        else:
            if action[0] in flags:
                camps_in_flags += 1
            
            if len(action) > 1 and action[1] in flags:
                camps_in_flags += 1
            
        check_camps_in_flags.append(camps_in_flags)
    '''
     #가장 많은 camp가 들어있거나 flag를 증가시키지 않는 index 탐색  
    min_increases = min(increase_camps_flags)
    max_camps = max(check_camps_in_flags)
    
    
    max_indices = [index for index, (value_ccinf, value_minincr) in enumerate(zip(check_camps_in_flags, increase_camps_flags)) if value_ccinf == max_camps and value_minincr == min_increases and index in valid_actions_idx]
    # 무작위 선택
    selected_action = random.choice(max_indices) if max_indices else 0
    return selected_action, action_matrix[selected_action]


def policy_Hold_At_Three_Flags_action(flags, is_boom):
    #print(f"lenflags : {len(flags)}")
    if is_boom == True:
        return 2
    
    if len(flags) == 3:
        return 1
    else:
        return 0

def policy_Hold_At_Three_Flags_action_v2(flags, is_boom, player_progress, turn_player_progress):#점수를 획득했다면 종료하는 조건 추가
    if is_boom == True:
        return 2
    
    if len(flags) == 3:
        return 1
    else:
        mapsize = np.array([3,5,7,9,11,13,11,9,7,5,3])
        got_score = False

        saved_score = 0
        turn_score = 0

        for idx, state in enumerate(player_progress):
            if state == mapsize[idx]:
                saved_score += 1
            
        for idx, state in enumerate(turn_player_progress):
            if state == mapsize[idx]:
                turn_score += 1


        got_score = turn_score > saved_score
        
        if got_score:
            return 1
        else:
            return 0

#####################################################

def policy_Score_Or_Bust_dice(action_matrix, action_masking, flags, turn_player_progress):
    valid_actions = []

    if action_masking[9] == 1:
        return 9
    
    for action in action_matrix:
        valid_actions.append(action if action_masking[action_matrix.index(action)] != 0 else None)

    check_scores = []
    mapsize = np.array([3,5,7,9,11,13,11,9,7,5,3])
    for action in valid_actions:
        score = 0
        if action is None:
            check_scores.append(score)
        else:
            for act in action:
                if turn_player_progress[act-2] + 1 == mapsize[act - 2]:
                    score += 1
    
        check_scores.append(score)

    max_score_action = max(check_scores)
    max_indices = [index for index, value in enumerate(check_scores) if value == max_score_action]
    
    selected_action = random.choice(max_indices) if max_indices else 0
    return selected_action

def policy_Score_Or_Bust_action(is_boom, player_progress, turn_player_progress, flags):
    if is_boom == True:
        return 2
    
    mapsize = np.array([3,5,7,9,11,13,11,9,7,5,3])
    got_score = False

    saved_score = 0
    turn_score = 0

    for idx, state in enumerate(player_progress):
        if state == mapsize[idx]:
            saved_score += 1
        
    for idx, state in enumerate(turn_player_progress):
        if state == mapsize[idx]:
            turn_score += 1


    got_score = turn_score > saved_score
    
    if got_score:
        return 1
    else:
        return 0
    

#####################################################

#gamefile의 learn_checkflags 코드를 가져옴
def extract_action_masking(flags, turn_player_progress, player_progress_opponent, action_matrix, map_combinations):
    action_masking = []
    mapsize = np.array([3,5,7,9,11,13,11,9,7,5,3])
    current_conquered_flag = np.zeros_like(mapsize)

    for idx, size in enumerate(mapsize):
        if size == turn_player_progress[idx]:
            current_conquered_flag[idx] = 1
        elif size == player_progress_opponent[idx]:
            current_conquered_flag[idx] = 2

    for idx, combinations in enumerate(action_matrix):
        if combinations is None:
            action_masking.append(0)
            continue
        
        if idx < 3:#combinations has 2 value. 
            if current_conquered_flag[combinations[0]-2] or current_conquered_flag[combinations[1]-2]:   
                action_masking.append(0)
                continue
            
            
            if len(flags) == 3:
                if combinations[0] in flags and combinations[1] in flags:
                    action_masking.append(1)
                else:
                    action_masking.append(0)
            elif len(flags) == 2:
                if combinations[0] == combinations[1]:
                    action_masking.append(1)
                elif combinations[0] in flags or combinations[1] in flags:
                    action_masking.append(1)
                else:
                    action_masking.append(0)
            else: # len(flags) <= 1
                action_masking.append(1)
            
        else: #idx >= 3 has only 1 value
            other_value = -1
            cur_idx = map_combinations[combinations[0]-2]

            #조합쌍 값을 찾음. 5,5같은 경우는 단일값 None처리를 했기에 반드시 대응하는 원소가 존재
            for val in range(len(map_combinations)):
                if cur_idx == map_combinations[val] and val + 2 != combinations[0]:
                    other_value = val + 2
            
            if current_conquered_flag[combinations[0]-2]:
                action_masking.append(0)
                continue

            if action_masking[cur_idx] == 1:
                action_masking.append(0)
                continue
            
            if len(flags) == 3: #단일값 가능 -> other_val이 flags 에 없다는 조건 필요
                if combinations[0] in flags and other_value not in flags:
                    action_masking.append(1)
                else:
                    action_masking.append(0)
            else: # 반대쌍값이 정복상태 -> 길이 2라도 단일값 가능, 아니면 둘다 flags에 해당x일때만 한 쪽 선택
                if current_conquered_flag[other_value-2] or (combinations[0] not in flags and other_value not in flags):
                    action_masking.append(1)
                else:
                    action_masking.append(0)
    check_impossible = 0
    for i in action_masking:
        check_impossible += i
    if check_impossible == 0:
        action_masking.append(1)
    else:
        action_masking.append(0)
        
    return action_masking