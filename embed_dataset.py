import os
import torch
from tools import builder
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from geometric_utils import *
from utils.config import cfg_from_yaml_file


# import Point-BERT and MLP and load weights
device = torch.device('cuda')

enc_checkpoint = torch.load('embedding_experiments/exp15/checkpoint', map_location=torch.device('cpu'))
encoder_head = enc_checkpoint['encoder_head'].to(device)

config = cfg_from_yaml_file('cfgs/ModelNet_models/PointTransformer.yaml')
model_config = config.model
pointbert = builder.model_builder(model_config)
weights_path = 'experiments/Point-BERT/Mixup_models/downloaded/Point-BERT.pth'
pointbert.load_model_from_ckpt(weights_path)
pointbert.to(device)

raw_path = '/home/alison/Clay_Data/Trajectory_Data/Aug24_Human_Demos/X'
processed_path = '/home/alison/Clay_Data/Trajectory_Data/Embedded_X'

n_trajs = 899
for i in tqdm(range(n_trajs)):
    os.mkdir(processed_path + '/Trajectory' + str(i))
    j = 0
    while exists(raw_path + '/Trajectory' + str(i) + '/state' + str(j) + '.npy'):
        
        # get the state data
        state = torch.from_numpy(np.load(raw_path + '/Trajectory' + str(i) + '/state' + str(j) + '.npy'))
        state = state.to(torch.float32)
        states = torch.unsqueeze(state, 0).to(device)

        # pass through Point-BERT
        tokenized_states = pointbert(states)
        embedded = encoder_head(tokenized_states)
        embedded = embedded.cpu().detach().numpy()

        # save the embedding as s_embedi.npy
        np.save(processed_path + '/Trajectory' + str(i) + '/s_embed' + str(j) + '.npy', embedded)

        if j != 0:
            action = np.load(raw_path + '/Trajectory' + str(i) + '/action' + str(j-1) + '.npy')
            np.save(processed_path + '/Trajectory' + str(i) + '/action' + str(j-1) + '.npy', action)

        j += 1