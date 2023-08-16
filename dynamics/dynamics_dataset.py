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
        
        # # for the human demos
        # if index % 60 == 0:
        #     next_state = np.expand_dims(np.load(self.root + '/Next_States/' + self.pcl_type + '_next_state' + str(index) + '.npy'), axis=0)
        # else:
        #     next_state = np.expand_dims(np.load(self.root + '/Next_States/' + self.pcl_type + '_state' + str(index) + '.npy'), axis=0)

        # # for the random actions
        next_state = np.expand_dims(np.load(self.root + '/Next_States/' + self.pcl_type + '_next_state' + str(index) + '.npy'), axis=0)

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
    

class DemoWordDataset(data.Dataset):
    """
    The dataset for the patch-level word classification-based dynamics
    model.
    """
    def __init__(self, root, pcl_type, dvae):
        self.root = root
        self.actions = np.load(self.root + '/action_normalized.npy').astype('float32')
        self.pcl_type = pcl_type
        self.sample_points_num = 1048
        self.npoints = 2048
        self.permutation = np.arange(self.npoints)
        self.dvae = dvae

        # get the encoding size of a state
        state = np.expand_dims(np.load(self.root + '/States/' + self.pcl_type + '_state' + str(0) + '.npy'), axis=0)
        state = state.squeeze()
        state = self.random_sample(state, self.sample_points_num)
        state = self.pc_norm(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.cuda()
        z_state, _, _, _ = self.dvae.encode(state)
        self.encoding_size = z_state.size()[1]

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
        return len(self.actions) # * self.encoding_size
    
    def __getitem__(self, index):
        a_idx = index # int(index / self.encoding_size)
        word_idx = (index % self.encoding_size) - 1

        action = self.actions[a_idx]
        state = np.expand_dims(np.load(self.root + '/States/' + self.pcl_type + '_state' + str(a_idx) + '.npy'), axis=0)
        
        if a_idx % 60 == 0:
            next_state = np.expand_dims(np.load(self.root + '/Next_States/' + self.pcl_type + '_next_state' + str(a_idx) + '.npy'), axis=0)
        else:
            next_state = np.expand_dims(np.load(self.root + '/Next_States/' + self.pcl_type + '_state' + str(a_idx) + '.npy'), axis=0)
        
        state = state.squeeze()
        state = self.random_sample(state, self.sample_points_num)
        state = self.pc_norm(state)

        next_state = next_state.squeeze()
        next_state = self.random_sample(next_state, self.sample_points_num)
        next_state = self.pc_norm(next_state)

        state = torch.from_numpy(state).float() #.unsqueeze(0)
        next_state = torch.from_numpy(next_state) #.float().unsqueeze(0)
        action = torch.from_numpy(action).float() #.unsqueeze(0)

        # state = state.cuda()
        # action = action.cuda()
        # next_state = next_state.cuda()

        # print("State Shape: ", state.size())
        # print("Action shape: ", action.size())
        # print("Next state shape: ", next_state.size())

        # z_state, _, center, _ = self.dvae.encode(state) 
        # gt_z_next_state, _, _, _ = self.dvae.encode(next_state)
        # ns_word = gt_z_next_state[:,word_idx,:] # NOTE: convert target to one-hot by looking up in codebook
        # target = self.dvae.codebook_onehot(ns_word).cuda()
        # group_center = center[:,word_idx,:]
        # vocab = z_state[:,word_idx,:]

        # print("\nVocab shape: ", vocab.size())
        # print("Group Center shape: ", group_center.size())
        # print("Target shape: ", target.size())
        
        # return vocab, group_center, action, target

        return state, next_state, action, word_idx
    

    # every self.encoding_size, index a new state, action, next state
        # (idx % self.encoding_size) - 1 should give the index of which word to get from state/next state encoding
        # int(idx / self.encoding_size) should give the index for state, action, next_state
