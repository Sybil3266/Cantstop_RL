import numpy as np
import random
import os
from collections import deque
from itertools import permutations, combinations
import copy
import torch

class Cantstop:
    def __init__ (self, players = 2, starting = None):
        self.players = players 
        self.score = np.zeros(self.players)
        self.mapsize = np.array([3,5,7,9,11,13,11,9,7,5,3])
        self.player_progress = {}
        self.flags = []
        self.game_finished = 0
        self.player_progress = {f'player{i+1}': np.zeros_like(self.mapsize) for i in range(players)}
        self.mode = 1 #play = 1 | run model = 2
        if starting is None:
            self.turn = 0
        else:
            self.turn = starting

                
    def __str__(self):
        return f"Player : {self.player}, Score : {self.score}"
    
    def __repr__(self):
        return f"Cantstop_call__repr__"
    ####
    def play(self):
        #make_gameplay_pipeline
        print(f"_____START_GAME_____")
        self.player = random.randint(1, self.players)
        print("\n\n--------------------------------\n\n")
        self.turn_player_progress = copy.copy(self.player_progress[f"player{self.player}"])
        self.conquered_flag = np.zeros_like(self.mapsize)
        play_turn = True
        while self.game_finished == 0:
            print(f"_____PLAYERS : {self.players}. CURRENT_PLAYER : {self.player} _____")
            print(f"_____CURRENT FLAGS : {self.flags} _____")
            dices, dice_sum_comb = self.dice()
            print(f"_____PLAYER_PROGRESS : {self.turn_player_progress} _____")
            print("\n")
            self.print_board()
            #self.print_board2()
            print("\n")
            print(f"_____DICE_RESULT : {dices} DICE_COMBS : {dice_sum_comb} _____")
            
            dice_comb_result, dice_sum_comb = self.check_flags(dice_sum_comb)
            if dice_comb_result == []:
                self.print_box("BOOM!")
                play_turn = False
                select_and_stop = 1
                self.turn_player_progress = copy.copy(self.player_progress[f"player{self.player}"])
                 
            while play_turn:
                select_comb_announce_string = f"_____SELECT DICE COMBS : {dice_comb_result} : "
                print("\n--------------------------------\n")
                
                user_input = input(select_comb_announce_string)
                if not user_input.isdigit():
                    print("_____ INVALID NUMBER _____")
                    continue
                
                select_and_stop = int(user_input)
                if select_and_stop > len(dice_comb_result) or select_and_stop <= 0:
                    print("_____ INVALID NUMBER _____")
                    continue

                for i in dice_comb_result[select_and_stop-1]:
                    if i not in self.flags:
                        self.flags.append(i)
                    #board has line 2 to 12. all board's list's index : 0 to 10 
                    self.turn_player_progress[i-2] += 1
                    
                break
            
            while play_turn:
                print("1: STOP AND SAVE THIS STATE")
                print("2: PLAY MORE DICE")
                select_and_stop = input("Choose an option (1/2): ")
                print("\n--------------------------------\n")
                if not select_and_stop.isdigit():
                    print("_____ INVALID NUMBER _____")
                    continue
                select_and_stop = int(select_and_stop)
                if select_and_stop not in [1, 2]:
                    print("____ INVALID NUMBER _____")
                    continue
                else:
                    break


            if select_and_stop == 1:
                self.flags = []
                self.player_progress[f"player{self.player}"] = copy.copy(self.turn_player_progress)
                for index in range(len(self.turn_player_progress)):
                    if self.turn_player_progress[index-2] == self.mapsize[index-2]:
                        self.conquered_flag[index-2] = self.player
                
                result = self.check_winpoint()
                if result != -1:
                    self.print_box(f"WINNER IS PLAYER {result}!")
                    self.game_finished = -1
                    continue
                
                self.player = self.player + 1 if self.player != self.players else 1
                self.turn_player_progress = copy.copy(self.player_progress[f"player{self.player}"])
                play_turn = True


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


        return result_comb, dice_sum_comb

    #THROW DICE + FIND COMPS
    def dice(self):
        dices = []
        self.dice_matrix = [0] * 33
        for i in range(4):
            dices.append(random.randint(1,6))

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
        #print(f"self.conquered_flag = {self.conquered_flag}")
        for i in range(len(self.score)):
            self.score[i] = 0

        for flag in self.conquered_flag:
            if flag != 0:
                self.score[flag-1] += 1
        
        
        for idx in range(len(self.score)):
            #print(f"self.score[{idx}] = {self.score[idx]}")
            if self.score[idx] >= 3:
                return idx+1
            
        return -1
    
    def reset(self):
        self.score = 0

        return f"__RESET__CANTSTOP__"
    
    def print_board(self):
        player_current = self.player
        player_opponent = self.player + 1 if self.player != self.players else 1
        progress_current = self.turn_player_progress
        progress_player = self.player_progress[f"player{player_current}"]
        progress_opponent = self.player_progress[f"player{player_opponent}"]

        max_size = self.mapsize.max()
        board = [[' ' for _ in range(max_size)] for _ in range(len(self.mapsize))]
        
        for i, row_size in enumerate(self.mapsize):
            start = (max_size - row_size) // 2
            end = start + row_size
            for j in range(start, end):
                board[i][j] = '*'


            pos2 = progress_player[i]
            pos3 = progress_opponent[i]

            if pos2 > 0:
                player_pos = end - pos2
                board[i][player_pos] = str(player_current)
            
            if pos3 > 0:
                player_pos = end - pos3
                if board[i][player_pos] == str(player_current):
                    board[i][player_pos] = "0"
                else:
                    board[i][player_pos] = str(player_opponent)


            pos = progress_current[i]
            if pos > 0:
                player_pos = end - pos
                board[i][player_pos] = "X"

            '''
            if pos > 0:
                player_pos = end - pos2
                player_pos2 = end - pos3
                if player_pos == player_pos2 and pos2 > 0:
                    board[i][player_pos] = "0"
                else:
                    board[i][player_pos] = str(player_current)
                    board[i][player_pos2] = str(player_opponent)
            '''
        # Rotate and print board
        rotated_board = list(zip(*board))
        for row in rotated_board:
            print(''.join(row))


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
        self.player = random.randint(1, self.players)
        self.turn_player_progress = copy.copy(self.player_progress[f"player{self.player}"])
        self.conquered_flag = np.zeros_like(self.mapsize)
        self.game_finished = False
        self.cumulative_rewards_dice = 0
        self.cumulative_rewards_action = 0

    def Learn_dice(self):
        #return dice combinations
        dices, dice_sum_comb = self.dice()
        action_matrix = []
        double_mark = []
        map_combinations = [-1] * 11
        

        for i in range(3):
            if i < len(dice_sum_comb):
                action_matrix.append(dice_sum_comb[i])
                map_combinations[dice_sum_comb[i][0] - 2] = i
                map_combinations[dice_sum_comb[i][1] - 2] = i
                if dice_sum_comb[i][0] == dice_sum_comb[i][1]:
                    double_mark.append(dice_sum_comb[i][0])
            else:
                action_matrix.append(None)

        #print(f"dices : {dices}")
        dices_sum = [sum(comb) if sum(comb) not in double_mark else 999 for comb in combinations(dices, 2)]
        #print(f"dices_sum before sort : {dices_sum}")


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
        '''
        #dices_sum = [x for x in dices_sum if x != 999]
        #dices_sum.sort()
        
        #preval = 999
        for i in range(len(dices_sum)):
            if preval == dices_sum[i]:
                dices_sum[i] = None    
            else:    
                preval = dices_sum[i]
            if dices_sum[i] != None:
                dices_sum[i] = tuple([dices_sum[i]])

        for val in dices_sum:
            action_matrix.append(val)
        #action_matrix.append(-1) # Can't play Action
        #action_matrix.append(0) # empty place
        print(f"dices_sum after sort : {temp_list}")
        print(f"action matrix : {action_matrix}")
        print(f"map_combinations : {map_combinations}")
        print(f"dice matrix : {self.dice_matrix}")
        print(f"action masking : {action_masking}")
        '''
        action_masking = self.Learn_check_flags(action_matrix, map_combinations)
        return action_matrix, action_masking

    def Learn_check_flags(self, action_matrix, map_combinations):
        action_masking = []

        for idx, combinations in enumerate(action_matrix):

            if combinations == None:
                action_masking.append(0)
                continue

            if idx < 3:#combinations has 2 value. 
                if self.conquered_flag[combinations[0]-2] or self.conquered_flag[combinations[1]-2]:   
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
                
                if self.conquered_flag[combinations[0]-2]:
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
                    if self.conquered_flag[other_value-2] or (combinations[0] not in self.flags and other_value not in self.flags):
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

            self.input_matrix = self.prepare_state(
                self.player_progress[f"player{player_current}"],
                self.player_progress[f"player{player_opponent}"],
                self.turn_player_progress,
                conquered_flag_input,
                self.dice_matrix
                )
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

            self.input_matrix = self.prepare_state(
                self.player_progress[f"player{player_current}"],
                self.player_progress[f"player{player_opponent}"],
                self.turn_player_progress,
                conquered_flag_input
                )
            return self.input_matrix
        
    
    def prepare_state(self, *lists):
        tensors = [torch.tensor(l, dtype=torch.float32) for l in lists]
        combined_tensor = torch.cat(tensors, dim=-1)
        return combined_tensor.unsqueeze(0)  # 배치 차원 추가
    
    def update_state_dice(self, action_index, action):#return rewards
        
        rewards = 0
        #print(f"action_index : {action_index} action : {action}. actiontype : {type(action)}")
        if action_index == 9:#state with no possible action
            #calculate_rewards
            rewards -= self.cumulative_rewards_dice * 1.1
            self.cumulative_rewards_dice = 0

            result, conquered_count = self.check_missed_points()
            if result == 1:
                rewards -= 200
            if conquered_count != 0:
                rewards -= 20 * conquered_count

            return -1, rewards, self.Learn_getState(1)
        else:
            #print(f"action : {action}")
            for i in action:
                if i not in self.flags:
                    self.flags.append(i)
                
                self.turn_player_progress[i-2] += 1
                rewards += 2 / (13 - (abs(7 - i+2) * 2))

            result, conquered_count = self.check_missed_points()#여기선 missed는 아니고, 이 행동으로 정복/승리 시 기대되는 점수
            if result == 1:
                rewards += 200
            if conquered_count != 0:
                rewards += 20 * conquered_count

            self.cumulative_rewards_dice += rewards
            return result, rewards, self.Learn_getState(1)
    

    def update_state_action(self, action):
        rewards = 0
        if action == 2:#state with no possible action
            #calculate rewards
            rewards -= self.cumulative_rewards_action * 1.1
            self.cumulative_rewards_action = 0
            result, conquered_count = self.check_missed_points()
            
            if result == 1:
                rewards -= 5000
            if conquered_count != 0:
                rewards -= 300 * conquered_count
            self.turn_player_progress = copy.copy(self.player_progress[f"player{self.player}"])
            result, conquered_count, state_for_update = self.change_turn(2)
            return -1, rewards, state_for_update
        else:
            if action == 0: #play more dice
                for idx, val in enumerate(self.turn_player_progress):
                    rewards += 2 / (13 - (abs(7 - idx+2) * 2)) * val
                self.cumulative_rewards_action += rewards
                state_for_update = self.Learn_getState(2)
                #rewards = 0
                return -1, rewards, state_for_update
            else: #stop and save this state
                rewards += self.cumulative_rewards_action
                self.cumulative_rewards_action = 0
                result, conquered_count, state_for_update = self.change_turn(2)

                if result == 1:
                    rewards += 200
                    self.game_finished = True
                if conquered_count != 0:
                    rewards += 20

                return result, rewards, state_for_update

    def check_missed_points(self):
        conquered_count = 0
        current_score = 0
        for index in range(len(self.turn_player_progress)):
            if self.turn_player_progress[index-2] == self.mapsize[index-2]:
                conquered_count += 1
        for i in self.turn_player_progress:
            if i == self.player:
                current_score += 1
        
        if conquered_count + current_score >= 3:
            return 1, conquered_count
        else:
            return -1, conquered_count

    def change_turn(self, n):
        self.flags = []
        self.player_progress[f"player{self.player}"] = copy.copy(self.turn_player_progress)
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
        self.turn_player_progress = copy.copy(self.player_progress[f"player{self.player}"])

        return result, conquered_count, state_for_replay
    
    def set_state(self, data_matrix):
        self.Learn_reset()
        data_list = data_matrix.toList()
        print(data_list)

