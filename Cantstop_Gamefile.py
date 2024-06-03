import numpy as np
import random
import os
from collections import deque, Counter
from itertools import permutations, combinations
import copy
import torch

class Cantstop:
    def __init__ (self, players = 2, starting = None):
        self.ams = []
        self.amx = []
        self.ccf = []
        self.players = players 
        self.score = np.zeros(self.players)
        self.mapsize = np.array([3,5,7,9,11,13,11,9,7,5,3])
        #self.mapsize = np.array([3,3,3,3,4,5,4,3,3,3,3]) #테스트용 작은 크기 map
        self.player_progress = {}
        self.flags = []
        self.game_finished = False
        self.player_progress = {f'player{i+1}': np.zeros_like(self.mapsize) for i in range(players)}
        self.mode = 1 #play = 1 | run model = 2
        self.already_conquered_point = np.zeros_like(self.mapsize)
        self.col_rewards = np.array([2 / cnt for cnt in self.mapsize])

    def __str__(self):
        return f"Player : {self.player}, Score : {self.score}"
    
    def __repr__(self):
        return f"Cantstop_call__repr__"
    ####
    def play_init(self):
        #make_gameplay_pipeline
        print(f"_____START_GAME_____")
        self.player = random.randint(1, self.players)
        print("\n\n--------------------------------\n\n")
        self.turn_player_progress = copy.deepcopy(self.player_progress[f"player{self.player}"])
        self.conquered_flag = np.zeros_like(self.mapsize)
        #self.playable = True
    
    def print_state(self):
        print(f"_____PLAYERS : {self.players}. CURRENT_PLAYER : {self.player} _____")
        print(f"_____CURRENT FLAGS : {self.flags} _____")
        print(f"_____PLAYER_PROGRESS : {self.turn_player_progress} _____")
        print("\n")
        self.print_board()
        print("\n")
        

    def play_game_turn(self):
        isBoom = False
        self.dices, self.dice_sum_comb = self.dice()
        print(f"_____DICE_RESULT : {self.dices} DICE_COMBS : {self.dice_sum_comb} _____")
        self.dice_comb_result, self.dice_sum_comb = self.check_flags(self.dice_sum_comb)
        if self.dice_comb_result == []:
            self.print_box("BOOM!")
            isBoom = True
            self.select_and_stop = 1
            self.player = self.player + 1 if self.player != self.players else 1
            self.turn_player_progress = copy.deepcopy(self.player_progress[f"player{self.player}"])
            self.flags = []
        return self.Learn_getState(1), isBoom
        
    def select_dice(self):
        while True:
            select_comb_announce_string = f"_____SELECT DICE COMBS : {self.dice_comb_result} : "
            print("\n--------------------------------\n")
            
            user_input = input(select_comb_announce_string)
            if not user_input.isdigit():
                print("_____ INVALID NUMBER _____")
                continue
            
            select_and_stop = int(user_input)
            if select_and_stop > len(self.dice_comb_result) or select_and_stop <= 0:
                print("_____ INVALID NUMBER _____")
                continue

            for i in self.dice_comb_result[select_and_stop-1]:
                if i not in self.flags:
                    self.flags.append(i)
                #board has line 2 to 12. all board's list's index : 0 to 10 
                self.turn_player_progress[i-2] += 1
                
            break

        return self.Learn_getState(2)

    def select_action(self):
        while True:
            print("1: PLAY MORE DICE")
            print("2: STOP AND SAVE THIS STATE")
            self.select_and_stop = input("Choose an option (1/2): ")
            print("\n--------------------------------\n")
            if not self.select_and_stop.isdigit():
                print("_____ INVALID NUMBER _____")
                continue
            self.select_and_stop = int(self.select_and_stop)
            if self.select_and_stop not in [1, 2]:
                print("____ INVALID NUMBER _____")
                continue
            else:
                break


        if self.select_and_stop == 2:
            self.flags = []
            self.player_progress[f"player{self.player}"] = copy.deepcopy(self.turn_player_progress)
            for index in range(len(self.turn_player_progress)):
                if self.turn_player_progress[index-2] == self.mapsize[index-2]:
                    self.conquered_flag[index-2] = self.player
            
            result = self.check_winpoint()
            if result != -1:
                self.print_box(f"WINNER IS PLAYER {result}!")
                self.game_finished = True
                
            
            self.player = self.player + 1 if self.player != self.players else 1
            self.turn_player_progress = copy.deepcopy(self.player_progress[f"player{self.player}"])
            #self.playable = True

        return self.game_finished

    def print_box(self, message):
        width = 20
        height = 10

        #half_width = width // 2
        half_height = height // 2 + 1

        print("+" + "-" * width + "+")

        for i in range(height):
            if i == half_height:
                spaces_before = (width - len(message)) // 2
                spaces_after = width - len(message) - spaces_before
                print("|" + " " * spaces_before + message + " " * spaces_after + "|")
            else:
                print("|" + " " * width + "|")

        print("+" + "-" * width + "+")    

        
    def check_flags(self, dice_sum_comb):
        result_comb = []
         
        for comb in dice_sum_comb:
            if len(self.flags) == 3:
                if comb[0] in self.flags and comb[1] in self.flags:
                    result_comb.append(comb)
                elif comb[0] in self.flags and self.conquered_flag[comb[0]-2] == 0:
                    result_comb.append((comb[0],))
                elif comb[1] in self.flags and self.conquered_flag[comb[1]-2] == 0:
                    result_comb.append((comb[1],))
            elif len(self.flags) == 2:
                if comb[0] in self.flags or comb[1] in self.flags:
                    result_comb.append(comb)
                elif comb[0] == comb[1]:
                    result_comb.append(comb)
                else:
                    if self.conquered_flag[comb[0]-2] == 0:
                        result_comb.append((comb[0],))
                    if self.conquered_flag[comb[1]-2] == 0:
                        result_comb.append((comb[1],))
            else: #len(self.flag) 0 or 1
                if self.conquered_flag[comb[0]-2] == 0 and self.conquered_flag[comb[1]-2] == 0:
                    result_comb.append(comb)
                elif self.conquered_flag[comb[0]-2] == 0:
                    result_comb.append((comb[0],))
                elif self.conquered_flag[comb[1]-2] == 0:
                    result_comb.append((comb[1],))
        result_comb.sort(key=lambda x:(len(x), x[0]))
        print(f"result comb : {result_comb}, dice sum comb : {dice_sum_comb}")
        return result_comb, dice_sum_comb

    #THROW DICE + FIND COMPS
    def dice(self):
        dices = []
        self.dice_matrix = [0] * 33
        for i in range(4):
            dices.append(random.randint(1,6))
        #dices = [1,1,3,3] #임의 조합 확인용
        dices.sort()
        #we can make 3 combinations
        comb1 = tuple([tuple([dices[0], dices[1]]), tuple([dices[2], dices[3]])])
        comb2 = tuple([tuple([dices[0], dices[2]]), tuple([dices[1], dices[3]])])
        comb3 = tuple([tuple([dices[0], dices[3]]), tuple([dices[1], dices[2]])])
        
        dice_comb = [comb1, comb2, comb3]

        dice_sum_comb = [ tuple([i[0][0] + i[0][1], i[1][0] + i[1][1]]) for i in dice_comb ]
        
        dice_sum_unique_comb = []
        i = 0
        for comb in dice_sum_comb:
            if comb not in dice_sum_unique_comb and tuple([comb[1], comb[0]]) not in dice_sum_unique_comb:
                dice_sum_unique_comb.append(comb)
                self.dice_matrix[i + comb[0]-2] += 1
                self.dice_matrix[i + comb[1]-2] += 1
                i += 11
        
        return dices, dice_sum_unique_comb
            
    def check_winpoint(self):
        for i in range(len(self.score)):
            self.score[i] = 0

        for flag in self.conquered_flag:
            if flag != 0:
                self.score[flag-1] += 1
        
        
        for idx in range(len(self.score)):
            if self.score[idx] >= 3:
                return idx+1
            
        return -1
    
    def reset(self):
        self.score = 0

        return f"__RESET__CANTSTOP__"
    

    def print_board2(self):

        player_current = self.player
        player_opponent = self.player + 1 if self.player != self.players else 1
        
        progress_current = self.turn_player_progress
        progress_player = self.player_progress[f"player{player_current}"]
        progress_opponent = self.player_progress[f"player{player_opponent}"]

        #max_size = 13
        max_size = self.mapsize.max() 
        #len(self.mapsize) = 11. mapsize[0~10]
        board = [[' ' for _ in range(max_size)] for _ in range(len(self.mapsize))]
        
        for i, row_size in enumerate(self.mapsize):
            """
            0 = 5
            1 = 4~6
            2 = 3~7
            """
            start = 5 - i
            end = 5 + i + 1
            if start <= 0:
                start = 0
            if end >= 12:
                end = 12
                
            print(f"start : {start} end : {end}")
            for j in range(start, end):
                board[i][j] = '*'

            pos = progress_current[i]
            if pos > 0:
                player_pos = max_size - row_size
                board[i][player_pos] = "X"

            pos2 = progress_player[i]
            pos3 = progress_opponent[i]
            if pos2 > 0:
                player_pos = max_size - row_size
                board[i][player_pos] = str(player_current)

            if pos3 > 0:
                player_pos = max_size - row_size
                if pos2 == pos3:
                    board[i][player_pos] = "X"
                else:
                    board[i][player_pos] = str(player_opponent)


        # Rotate and print board
        #rotated_board = list(zip(*board))
        for row in board:
            print(''.join(row))


    def Learn_reset(self):
        #scripts for Learning Model
        self.score = np.zeros(self.players)
        self.player_progress = {}
        self.flags = []
        self.game_finished = False
        self.player_progress = {f'player{i+1}': np.zeros_like(self.mapsize) for i in range(self.players)}

        self.player = random.randint(1, self.players)
        self.turn_player_progress = copy.deepcopy(self.player_progress[f"player{self.player}"])
        self.conquered_flag = np.zeros_like(self.mapsize)
        self.cumulative_rewards_dice = 0
        self.cumulative_rewards_action = 0


    def Learn_dice(self):
        #return dice combinations
        dices, dice_sum_comb = self.dice()
        action_matrix = []
        double_mark = []
        map_combinations = [-1] * 11
        #print(f"dices : {dices} ")
        #print(f"dice sum comb : {dice_sum_comb}")
        for i in range(3):
            if i < len(dice_sum_comb):
                action_matrix.append(dice_sum_comb[i])
                map_combinations[dice_sum_comb[i][0] - 2] = i
                map_combinations[dice_sum_comb[i][1] - 2] = i
                if dice_sum_comb[i][0] == dice_sum_comb[i][1]:
                    double_mark.append(dice_sum_comb[i][0])
            else:
                action_matrix.append(None)

        dices_sum = [sum(comb) if sum(comb) not in double_mark else 999 for comb in combinations(dices, 2)]


        # 정렬 및 중복 제거를 위한 임시 리스트
        temp_list = [x for x in dices_sum if x != 999 and x is not None]
        temp_list.sort()

        preval = -1
        sorted_list = []
        for val in temp_list:
            if preval != val:
                sorted_list.append(tuple([val]))
                preval = val
        # 원래 action_matrix에 정렬된 리스트 추가
        action_matrix.extend(sorted_list)
        
        for i in range(9 - len(action_matrix)):
            action_matrix.append(None)
        #print(f"action matrix : {action_matrix}")
        action_masking = self.Learn_check_flags(action_matrix, map_combinations)
        #print(f"action masking : {action_masking}")
        return action_matrix, action_masking

    def Learn_check_flags(self, action_matrix, map_combinations):
        action_masking = []
        
        #진행중인 턴에 정복을 했을 경우, 해당 flag는 행동이 불가능하기 때문에 임시 list가 추가로 필요
        #self.conquered_flags는 행동을 종료할 때만 갱신해야함
        
        current_conquered_flag = np.zeros_like(self.mapsize)
        player_opponent = self.player + 1 if self.player != self.players else 1
        for idx, size in enumerate(self.mapsize):
            if size == self.turn_player_progress[idx]:
                current_conquered_flag[idx] = self.player
            elif size == self.player_progress[f"player{player_opponent}"][idx]:
                current_conquered_flag[idx] = player_opponent
        self.ccf = current_conquered_flag
        #print(f"self.flags : {self.flags}")
        for idx, combinations in enumerate(action_matrix):

            if combinations is None:
                action_masking.append(0)
                continue

            if idx < 3:#combinations has 2 value. 
                if current_conquered_flag[combinations[0]-2] or current_conquered_flag[combinations[1]-2]:   
                    action_masking.append(0)
                    continue

                if len(self.flags) == 3:
                    if combinations[0] in self.flags and combinations[1] in self.flags:
                        action_masking.append(1)
                    else:
                        action_masking.append(0)
                elif len(self.flags) == 2:
                    if combinations[0] == combinations[1]:
                        action_masking.append(1)
                    elif combinations[0] in self.flags or combinations[1] in self.flags:
                        action_masking.append(1)
                    else:
                        action_masking.append(0)
                else: # len(self.flags) <= 1
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
                
                if len(self.flags) == 3: #단일값 가능 -> other_val이 flags 에 없다는 조건 필요
                    if combinations[0] in self.flags and other_value not in self.flags:
                        action_masking.append(1)
                    else:
                        action_masking.append(0)
                else: # 반대쌍값이 정복상태 -> 길이 2라도 단일값 가능, 아니면 둘다 flags에 해당x일때만 한 쪽 선택
                    if current_conquered_flag[other_value-2] or (combinations[0] not in self.flags and other_value not in self.flags):
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
        self.amx = action_matrix
        self.ams = action_masking
        return action_masking
    
    def Learn_getState(self, datatype):
        self.input_matrix = []
        if datatype == 1: #data for SelectDiceCombDQN
            '''
            input data 
            self.player_progress = 22
            self.turn_player_progress = 11 
            self.conquered_flag = 11
            //self.dices
            //self.action_matrix
            self.dice_matrix = 33
            input의 dim에는 플레이어의 기준이 없고, 내 상태 - 상대 상태 위치를 맞춰서 넣는다.
            self.conquered_flag에는 플레이어 번호의 정복된 상태가 들어있기 때문에, 본인을 1, 상대를 2로 변환시켜 동일하게 처리

            이거도 2차원으로 바꿔야지..
        
            '''
            player_current = self.player
            player_opponent = self.player + 1 if self.player != self.players else 1
            conquered_flag_input = []
            for val in self.conquered_flag:
                if val == player_current:
                    conquered_flag_input.append(1)
                elif val == player_opponent:
                    conquered_flag_input.append(2)
                else:
                    conquered_flag_input.append(0)
            '''
            self.input_matrix = self.prepare_state(
                self.player_progress[f"player{player_current}"],
                self.player_progress[f"player{player_opponent}"],
                self.turn_player_progress,
                conquered_flag_input,
                self.dice_matrix
                )
            '''
            self.input_matrix = self.prepare_state(dice_or_action=1, conquered_flag_input=conquered_flag_input)

            return self.input_matrix
        else: #data for SelectAdditionalActionDQN
            player_current = self.player
            player_opponent = self.player + 1 if self.player != self.players else 1
            conquered_flag_input = []

            for val in self.conquered_flag:
                if val == player_current:
                    conquered_flag_input.append(1)
                elif val == player_opponent:
                    conquered_flag_input.append(2)
                else:
                    conquered_flag_input.append(0)
            '''
            self.input_matrix = self.prepare_state(
                self.player_progress[f"player{player_current}"],
                self.player_progress[f"player{player_opponent}"],
                self.turn_player_progress,
                conquered_flag_input
                )
            '''
            self.input_matrix = self.prepare_state(dice_or_action=2, conquered_flag_input=conquered_flag_input)
            return self.input_matrix
        
    '''
    def prepare_state(self, *lists):
        tensors = [torch.tensor(l, dtype=torch.float32) for l in lists]
        combined_tensor = torch.cat(tensors, dim=-1)
        return combined_tensor.unsqueeze(0)  # 배치 차원 추가
    '''
    
    def prepare_state(self, dice_or_action, conquered_flag_input):
        player_current = self.player
        player_opponent = self.player + 1 if self.player != self.players else 1
        
        conquered_flag_input = np.array([
            1 if val == player_current else 2 if val == player_opponent else 0
            for val in self.conquered_flag
        ], dtype=np.float32)
        
        if dice_or_action == 1:
            data_arrays = [
                np.array(self.player_progress[f"player{player_current}"], dtype=np.float32),
                np.array(self.player_progress[f"player{player_opponent}"], dtype=np.float32),
                np.array(self.turn_player_progress, dtype=np.float32),
                conquered_flag_input,
                np.array(self.dice_matrix[0:11], dtype=np.float32),
                np.array(self.dice_matrix[11:22], dtype=np.float32),
                np.array(self.dice_matrix[22:33], dtype=np.float32)
            ]
        else:
            data_arrays = [
                np.array(self.player_progress[f"player{player_current}"], dtype=np.float32),
                np.array(self.player_progress[f"player{player_opponent}"], dtype=np.float32),
                np.array(self.turn_player_progress, dtype=np.float32),
                conquered_flag_input
            ]

        combined_array = np.concatenate(data_arrays)
        if dice_or_action == 1:
            self.input_matrix = torch.tensor(combined_array).view(7, 11).unsqueeze(0)
        else:
            self.input_matrix = torch.tensor(combined_array).view(4, 11).unsqueeze(0)
            
        #self.input_matrix = torch.tensor(combined_array)#.view(1, -1)
        #print(f"inputmat shape : {np.shape(self.input_matrix)}")
        return self.input_matrix
    
    #dice action 반영
    def update_state_dice(self, action_index, action):#return rewards
        #print(f"update state dice : action index : {action_index}, action : {action}")
        rewards = 0
        #print(f"action_index : {action_index} action : {action}. actiontype : {type(action)}")
        if action_index == 9:#state with no possible action
            #calculate_rewards
            rewards -= self.cumulative_rewards_dice * 1.1
            self.cumulative_rewards_dice = 0

            result, conquered_count = self.check_missed_points(-1)
            if result == 1:
                rewards -= 200
            if conquered_count != 0:
                rewards -= 20 * conquered_count

            return -1, rewards, self.Learn_getState(1)
        else:
            checker = False
            #오류 1 : 한번에 2칸을 뚫어서 오류가 생김
            '''
            if len(action) > 1 and action[0] == action[1]:
                if self.turn_player_progress[action[0] - 2] + 1 == self.mapsize[action[0] - 2]:
                    print(f"\n\n --------------------------------------------------------")
                    print(f"action is : {action}")
                    print(f"amx : {self.amx}")
                    print(f"ams : {self.ams}")
                    print(f"flg : {self.flags}")
                    print(f"ccf : {self.ccf}")
                    #print(f"TPP idx : {idx} val : {val}, maplim : {self.mapsize[idx]}")
                    self.translate_state(self.Learn_getState(1), 1, True)
                    print(f"----------------------------------------------------------\n\n")
            '''
            #우선 오류 1에 해당하는 조건만 처리 후 다른버그 탐색
            '''            
            if len(action) > 1 and action[0] == action[1]:
                if self.turn_player_progress[action[0] - 2] + 1 == self.mapsize[action[0] - 2]:
                    if action[0] not in self.flags:
                        self.flags.append(action[0])
                    
                    self.turn_player_progress[action[0]-2] += 1
                    rewards += 2 / (13 - (abs(7 - action[0]+2) * 2))
                else:
                    for i in action:
                        if i not in self.flags:
                            self.flags.append(i)
                    if self.turn_player_progress[i - 2] + 1 <= self.mapsize[i - 2]:  # 추가된 조건
                        self.turn_player_progress[i - 2] += 1
                        rewards += 2 / (13 - (abs(7 - i + 2) * 2))                        
            else:
                #print(f"action is : {action}")
                #print(f"flags before input : {self.flags}")
                for i in action:
                    if i not in self.flags:
                        self.flags.append(i)
                        #print(f"append {i} in self.flags : {self.flags}")
                    #else:
                        #print(f"passed appending value : {i}, self.flags : {self.flags}")
                    self.turn_player_progress[i-2] += 1
                    rewards += 2 / (13 - (abs(7 - i+2) * 2))
            '''
            for i in action:
                if i not in self.flags:
                    self.flags.append(i)
                    #print(f"append flags : {i}")
                if self.turn_player_progress[i - 2] + 1 <= self.mapsize[i - 2]:  # 추가된 조건
                    self.turn_player_progress[i - 2] += 1
                    rewards += self.col_rewards[i-2]                       
 
            #이유는 모르겠는데 action_masking은 정상 작동하는데 action_index가 9로 처리되지 않는 경우
            for idx, val in enumerate(self.turn_player_progress):

                if self.mapsize[idx] < val:
                    print(f"action index : {action_index}")
                    print(f"action is : {action}")
                    print(f"amx : {self.amx}")
                    print(f"ams : {self.ams}")
                    print(f"flg : {self.flags}")
                    print(f"ccf : {self.ccf}")
                    print(f"TPP idx : {idx} val : {val}, maplim : {self.mapsize[idx]}")
                    self.translate_state(self.Learn_getState(1), 1, True)
                    return -1


            result, conquered_count = self.check_missed_points(1)
            if result == 1:
                rewards += 200
            if conquered_count != 0:
                rewards += 30 * conquered_count
            #print(f"result : {result}, conquered count : {conquered_count}, total rewards : {rewards}")
            self.cumulative_rewards_dice += rewards
            return result, rewards, self.Learn_getState(1)
    
    #action반영
    def update_state_action(self, action):
        rewards = 0
        if action == 2:#state with no possible action
            #calculate rewards
            rewards -= self.cumulative_rewards_action * 1.1
            self.cumulative_rewards_action = 0
            result, conquered_count = self.check_missed_points(-1)
            
            if result == 1:
                rewards -= 5000
            if conquered_count != 0:
                rewards -= 300 * conquered_count
            self.turn_player_progress = copy.deepcopy(self.player_progress[f"player{self.player}"])
            #print(f"curpl //boom : {self.player}")
            result, conquered_count, state_for_update = self.change_turn(2)
            #print(f"after changeturn // boom : {self.player}")
            return -1, rewards, state_for_update
        else:
            if action == 0: #play more dice
                for idx, val in enumerate(self.turn_player_progress):
                    rewards += self.col_rewards[idx] * val
                self.cumulative_rewards_action += rewards
                state_for_update = self.Learn_getState(2)
                #rewards = 0
                return -1, rewards, state_for_update
            else: #stop and save this state
                rewards += self.cumulative_rewards_action
                self.cumulative_rewards_action = 0
                #print(f"curpl : {self.player}")

                result, conquered_count, state_for_update = self.change_turn(2)
                #print(f"after changeturn : {self.player}")

                if result == 1:
                    rewards += 200
                    self.game_finished = True
                if conquered_count != 0:
                    rewards += 30

                return result, rewards, state_for_update

    
    def check_missed_points(self, type): # 이번 턴에 놓친 점수를 계산한 뒤, conquered - current_score -> 손실 점수의 크기
        #type -1 => boomed situation, 1 => normal situation        
        conquered_count = 0
        current_score = 0
        for index in range(len(self.turn_player_progress)):
 
            if self.turn_player_progress[index-2] == self.mapsize[index-2]:
                if type == 1:#게임 진행중엔 conquered_count를 1번씩만 리턴 -> 그래야 끝점 도달시에만 점수
                    if self.already_conquered_point[index-2] == 0:
                        conquered_count += 1
                        self.already_conquered_point[index-2] = 1
                else: #type is -1
                    #터졌을 땐 손실점수 확인 목적이니 conquered_count 종합해서 반환
                    conquered_count += 1

        for i in self.player_progress[f"player{self.player}"]:
            if i == self.mapsize[index - 2]:
                current_score += 1
        #게임이 끝날 상황인지 확인 + 해당 행동(진행중) / 해당 턴(터짐)의 획득 점수 확인
        if conquered_count >= 3:
            return 1, conquered_count - current_score
        else:
            return -1, conquered_count - current_score
        
    #턴 전환 + 승리 여부 확인(학습용)
    def change_turn(self, n):
        self.flags = []
        self.player_progress[f"player{self.player}"] = copy.deepcopy(self.turn_player_progress)
        conquered_count = 0
        for index in range(len(self.turn_player_progress)):
            if self.turn_player_progress[index-2] == self.mapsize[index-2]:
                self.conquered_flag[index-2] = self.player
                conquered_count += 1
        if n == 1:
            state_for_replay = self.Learn_getState(1)
        else:
            state_for_replay = self.Learn_getState(2)
            
        result = self.check_winpoint() #If there is a winner, it returns player's number
        self.player = self.player + 1 if self.player != self.players else 1
        self.turn_player_progress = copy.deepcopy(self.player_progress[f"player{self.player}"])

        return result, conquered_count, state_for_replay
    
    def set_state(self, data_matrix):
        self.Learn_reset()
        data_list = data_matrix.toList()
        print(data_list)

    #7*11 형태의 state를 여러개의 변수로 다시 변환
    def translate_state(self, state, dtype, print_data):
        '''
            DTYPE1 = DICE
            input data 
            self.player_progress = 22
            self.turn_player_progress = 11 
            self.conquered_flag = 11
            //self.dices
            //self.action_matrix
            self.dice_matrix = 33
            input의 dim에는 플레이어의 기준이 없고, 내 상태 - 상대 상태 위치를 맞춰서 넣는다.
            self.conquered_flag에는 플레이어 번호의 정복된 상태가 들어있기 때문에, 본인을 1, 상대를 2로 변환시켜 동일하게 처리

            action에 입력되는 state는 dice_matrix를 제외한 1*44의 array가 input
        '''
    
        if dtype == 1:#state for dice
            state = state.reshape(7,11)
            player_progress_current = state[0, :]  # 첫 번째 행
            player_progress_opponent = state[1, :]  # 두 번째 행
            turn_player_progress = state[2, :]  # 세 번째 행
            conquered_flag = state[3, :]  # 네 번째 행
            possible_actions1 = state[4, :]
            possible_actions2 = state[5, :]
            possible_actions3 = state[6, :]
            actions = [possible_actions1, possible_actions2, possible_actions3]

            if print_data == True:
                err = False
                print(f"------------data in game--------------------")
                print(f"current player is : {self.player}")
                print(f"self.player_progress [ {self.player} ] current : {self.player_progress[f'player{self.player}']}")
                player_opponent = self.player + 1 if self.player != self.players else 1
                print(f"self.player_progress [ {player_opponent} ] opponent : {self.player_progress[f'player{player_opponent}']}")
                print(f"self.turn_player_progress : {self.turn_player_progress}")
                print(f"current flags : {self.flags}")
                print(f"ams : {self.ams}")
                print(f"amx : {self.amx}")
                
                for idx, val in enumerate(self.turn_player_progress):
                    if self.player_progress[f'player{self.player}'][idx] > val:
                        print(f"something looks gonna wrong in game class")
                        err = True
                print(f"------------data in state-------------------")
                print(f"Player_Progress_Current :  {player_progress_current}")
                print(f"Player_Progress_opponent : {player_progress_opponent}")
                print(f"turn_player_progress :     {turn_player_progress}")
                print(f"conquered_flag :           {conquered_flag}")

                for idx, val in enumerate(turn_player_progress):
                    if player_progress_current[idx] > val:
                        print(f"something looks gonna wrong in state data")
                        err = True

                if err == True:
                    print(f"err")
                    print(f"{self.player_progress[2000]}")
            # 가능한 행동들에 대해 반복
                for i, action in enumerate(actions, 1):
                    actiontuple = []
                    for idx in range(len(action)):
                        if action[idx] == 1:
                            actiontuple.append(idx+2)
                        elif action[idx] == 2:
                            actiontuple.append(idx+2)
                            actiontuple.append(idx+2)

                    actiontuple = tuple(actiontuple)
                    print(f"action{i} :                  {action} print by tuple : {actiontuple}")

            return player_progress_current, player_progress_opponent, turn_player_progress, conquered_flag, actions 
        else: #state for action
            state = state.reshape(4,11)
            player_progress_current = state[0, :]  # 첫 번째 행
            player_progress_opponent = state[1, :]  # 두 번째 행
            turn_player_progress = state[2, :]  # 세 번째 행
            conquered_flag = state[3, :]  # 네 번째 행
            if print_data == True:
                print(f"Player_Progress_Current : {player_progress_current}")
                print(f"Player_Progress_opponent : {player_progress_opponent}")
                print(f"turn_player_progress : {turn_player_progress}")
                print(f"conquered_flag : {conquered_flag}")
            return player_progress_current, player_progress_opponent, turn_player_progress, conquered_flag

    def are_lists_equal(self, list1, list2):
        if not list1 and not list2:
            return True
        return Counter(list1) == Counter(list2)

    def check_error(self):
        isError = False
        isFinished = False
        #Flags의 길이 초과
        if len(self.flags) > 3:
            print(f"ERROR : flags overflow {self.flags}")
            isError = True
        
        checkflags = []
        player_current = self.player
        player_opponent = self.player + 1 if self.player != self.players else 1
        progress_current = self.turn_player_progress
        progress_player = self.player_progress[f"player{player_current}"]
        progress_opponent = self.player_progress[f"player{player_opponent}"]

        #player_progress 오류 
        for idx, val in enumerate(self.turn_player_progress):
            if val - progress_player[idx] < 0:
                print("ERROR : turn_player_progress doesn't match")
                print(f"player : {self.player}")
                print(f"turn_player_progress : {self.turn_player_progress}")
                print(f"progress_player      : {progress_player}")
                print(f"progress_opponent    : {progress_opponent}")
                isError = True
            elif val - progress_player[idx] > 0:
                checkflags.append(idx+2)

        #flags의 원소 오류
        if not self.are_lists_equal(checkflags, checkflags):
            print("ERROR : flags are not equal")
            print(f"checkflags : {checkflags}")
            print(f"self.flags : {self.flags}")
            isError = True


        #mapsize 초과
        for idx, val in enumerate(self.turn_player_progress):
            if self.mapsize[idx] < val:
                print("ERROR : OVERFLOW IN MAPSIZE")
                print(f"self.turn_player_progress : {self.turn_player_progress}")
                isError = True

        #게임이 종료되지 않는 경우
        check_score = 0 
        for idx,val in enumerate(progress_player):
            if self.mapsize[idx] == val:
                check_score += 1

        if check_score >= 3:
            #print(f"GAME Finished BUT NOT RETURN : WINNER IS {player_current} // CURRENT")
            isFinished = True

        check_score = 0 
        for idx,val in enumerate(progress_opponent):
            if self.mapsize[idx] == val:
                check_score += 1

        if check_score >= 3:
            #print(f"GAME Finished BUT NOT RETURN : WINNER IS {player_opponent} // OPPONENT")
            isFinished = True

        return isError, isFinished