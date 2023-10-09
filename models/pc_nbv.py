import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import Feature_Extraction, SelfAttention, normalize_point_batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym

class Encoder(nn.Module):
    def __init__(self, in_channel=561):
        super(Encoder, self).__init__()
        self.feature_extraction = Feature_Extraction()
        self.bn1 = nn.BatchNorm1d(264)

        self.attention_unit = SelfAttention(in_channel)
        self.bn2 = nn.BatchNorm1d(561)

        self.conv1 = nn.Conv1d(561, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)


    def forward(self, inputs, view_state):
        inputs = normalize_point_batch(inputs)      #(B, 3, N)
        n = inputs.size()[2]

        x = self.feature_extraction(inputs)         #(B, 264, N)
        x = self.bn1(x)

        g = torch.max(x, dim=2, keepdim=True)[0]    #(B, 264, 1)
        g = g.repeat(1, 1, n)                       #(B, 264, N)

        vi = view_state.unsqueeze(2).repeat(1, 1, n)
        x = torch.cat([x, g, vi], dim = 1)          #(B, 561, N)   561 = 264 + 264 + 33

        x = self.attention_unit(F.relu(x))          #(B, 561, N)
        x = self.bn2(x)

        x = F.relu(self.bn3(self.conv1(x)))         #(B, 1024, N)
        x = self.bn4(self.conv2(x))                 #(B, 1024, N)

        v = torch.max(x, dim = -1)[0]               #(B, 1024)

        return v

class Decoder(nn.Module):
    def __init__(self, views=33):
        super(Decoder, self).__init__()
        self.views = views

        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, self.views)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))       #(B, 1024)
        x = F.relu(self.bn2(self.linear2(x)))       #(B, 512)
        x = F.relu(self.bn3(self.linear3(x)))       #(B, 256)

        v = self.linear4(x)                         #(B, 33)

        return v


class AutoEncoder(nn.Module):
    def __init__(self, views=33):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(views)
    
    def forward(self, x, viewstate):
        x = self.encoder(x, viewstate)
        v = self.decoder(x)
        return x, v


class CustomNeuralNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # views_num = observation_space["view_state"].shape[0]
        self.encoder = Encoder()
        self.decoder = Decoder(features_dim)
    
    def forward(self, observations: gym.spaces.Dict):
        x = observations["current_point_cloud"]
        viewstate = observations["view_state"]
        x = self.encoder(x, viewstate)
        v = self.decoder(x)
        return v


if __name__ == "__main__":
    pcs = torch.rand(16, 3, 2048)
    viewstate = torch.rand(16, 33)
    encoder = Encoder()
    x = encoder(pcs, viewstate)
    print(x.size())

    decoder = Decoder()
    decoder(x)
    v = decoder(x)
    print(v.size())

    ae = AutoEncoder()
    x, v = ae(pcs, viewstate)
    print(x.size(), v.size())

    loss = nn.MSELoss()
    total = sum([loss(param, torch.zeros_like(param)) for param in ae.encoder.parameters()])

    print('+ Number of params: %.2f'%(total))

