import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

'''
전체적인 학습 과정

1. init -> DQNClass.__init__()  
2. state detection -> Gamefile will give data to model
3. DQN input -> return all action's q_value
4. Action Masking -> use one-hot encoding for check active actions
5. select actions -> usually using epsilon-greedy policy. epsilon value will be reduced by time goes.
6. get rewards -> give rewards by calculation
7. save data in Replay Memory -> make learning more efficient by reusing memory
8. update q_value with rewards

''' 

'''
input dimension = 1*77/ 1*44인 코드
class SelectDiceCombDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SelectDiceCombDQN, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class SelectAdditionalActionDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelectAdditionalActionDQN, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
    

'''
#input dim을 2차원으로 바꿈 -> 7*11, 4*11로 변환

class SelectDiceCombDQN(nn.Module):

    def __init__(self, input_channels = 1, input_height = 7, input_width = 11, output_dim = 10):
        super(SelectDiceCombDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64 * input_height * input_width, 512)
        self.head = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.head(x)

class SelectAdditionalActionDQN(nn.Module):
    def __init__(self, input_channels = 1, input_height = 7, input_width = 11, output_dim = 3):
        super(SelectAdditionalActionDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64 * input_height * input_width, 512)
        self.head = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.head(x)
    
'''
334
351
input으로 이용되는 변수 수정
while로 게임이 진행되는 과정을 외부로 빼내고 수정해야함
그렇다면 어떻게 바꿔야하지

기존 코드일 경우 게임 종료까지 while -> self.game_finished return해서 값 확인으로 변경
주사위 조합을 return하고, 이를 return해주어야 state 전달이 가능
진행 후 상태를 actionDQN에 입력해야 추가/중지에 대한 가치가 나옴
이후 턴이 넘어가고 game_finished에 대한 check 진행

분리할 구간을 정하자

초기화 / 주사위를 던지고 결과를 return한다 / 조합을 선택한 후 상태를 저장하고 바로 actionDQN에 넘긴다 / 종료 상태를 확인하고 보상을 정산한다.
다시 주사위를 던지고 결과를 return한다
서로 학습하는 경우에 대해 어떻게 차이를 둘지만 확인을 해보자
'''