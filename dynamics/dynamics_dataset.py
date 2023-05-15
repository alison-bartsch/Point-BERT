import os
import torch
import pickle as pkl
import numpy as np
import torch.utils.data as data

class DemoActionDataset(data.Dataset):
    """
    """

    def __init__(self, root, pcl_type):
        self.root = root
        self.actions = np.load(self.root + '/action_normalized.npy').astype('float32')
        self.pcl_type = pcl_type
        self.sample_points_num = 1048
        self.npoints = 2048
        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        action = self.actions[index]
        state = np.expand_dims(np.load(self.root + '/States/' + self.pcl_type + '_state' + str(index) + '.npy'), axis=0)
        

        if index % 60 == 0:
            next_state = np.expand_dims(np.load(self.root + '/Next_States/' + self.pcl_type + '_next_state' + str(index) + '.npy'), axis=0)
        else:
            next_state = np.expand_dims(np.load(self.root + '/Next_States/' + self.pcl_type + '_state' + str(index) + '.npy'), axis=0)
        
        state = state.squeeze()
        state = self.random_sample(state, self.sample_points_num)
        state = self.pc_norm(state)

        next_state = next_state.squeeze()
        next_state = self.random_sample(next_state, self.sample_points_num)
        next_state = self.pc_norm(next_state)

        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        action = torch.from_numpy(action).float()
        
        return state, next_state, action
    