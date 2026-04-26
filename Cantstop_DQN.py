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

---------------------------------------------------------------------------
Channel-separated version (Conv1d)

기존 구조:
    input: (batch, 1, H, 11)        -- 7개(혹은 4개) semantic row가
                                       한 채널 안에 섞여서 2D CNN으로 처리됨
    conv2d kernel 5x5 가 row-axis 를 가로지르면서 의미가 다른 행끼리 섞임

변경 구조:
    input: (batch, C, 11)           -- 각 semantic row 를 독립 채널로 분리
        dice  -> C = 7 (player/opponent/turn progress, conquered, 3 possible dice row)
        action-> C = 4 (player/opponent/turn progress, conquered)
    Conv1d 가 "column 축" 으로만 convolve -> 열(기둥) 간의 지역 관계만 학습하고,
    채널 간 정보는 filter weight 로 선형 결합 됨. 행이 섞이지 않으므로
    "이 숫자가 어떤 의미의 값인지" 를 모델이 구조적으로 인식할 수 있다.
---------------------------------------------------------------------------
'''


class SelectDiceCombDQN(nn.Module):
    """Dice combination Q-network. Input: (batch, input_channels=7, input_width=11)."""

    def __init__(self, input_channels=7, input_width=11, output_dim=10):
        super(SelectDiceCombDQN, self).__init__()
        self.input_channels = input_channels
        self.input_width = input_width

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128 * input_width, 512)
        self.head = nn.Linear(512, output_dim)

    def forward(self, x):
        # x: (batch, input_channels, input_width)
        # 뒤쪽 호환: (batch, 1, C, W) 형태로 오면 squeeze
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.head(x)


class SelectAdditionalActionDQN(nn.Module):
    """Turn-action Q-network. Input: (batch, input_channels=4, input_width=11)."""

    def __init__(self, input_channels=4, input_width=11, output_dim=3):
        super(SelectAdditionalActionDQN, self).__init__()
        self.input_channels = input_channels
        self.input_width = input_width

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64 * input_width, 256)
        self.head = nn.Linear(256, output_dim)

    def forward(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.head(x)
