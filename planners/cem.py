import os
import time
import torch
import json
import random
import argparse
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm
from planners.robot_utils import *
from planners.plan_geometric_utils import * 
from os.path import join
from pytorch3d.loss import chamfer_distance
from planners.geometric_sampler import GeometricSampler

class CEM():
    def __init__(self, device, dvae, feature_dynamics_network, planning_horizon, n_actions, action_dim, sampler='random'):
        self.device = device
        self.dvae = dvae
        self.feature_dynamics_network = feature_dynamics_network
        self.planning_horizon = planning_horizon
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.sampler = sampler

    def normalize_pcl(self, pcl):
        """
        """
        centroid = np.mean(pcl, axis=0)
        pcl = pcl - centroid
        m = np.max(np.sqrt(np.sum(pcl**2, axis=1)))
        pcl = pcl / m
        return pcl

    def dynamics_model(self, state, action):

        """
        Assumes unnormalized state and normalized action as inputs in numpy format. 
        """

        # going from state and action to next state with trained models
        # print("State Mean: ", np.mean(state, axis=0))
        # print("Normalized State Mean: ", np.mean(self.normalize_pcl(state), axis=0))
        normalized_state = torch.from_numpy(self.normalize_pcl(state)).float().unsqueeze(0).cuda()
        normalized_action = torch.from_numpy(action).cuda()

        z_states, neighborhood, center, logits = self.dvae.encode(normalized_state) #.to(device)
        
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        ns_center_unnormalized = predict_centroid_dynamics(state, action)

        normalization_centroid = np.mean(state.detach().numpy(), axis=0)
        ns_center = ns_center_unnormalized - normalization_centroid
        m = np.max(np.sqrt(np.sum(ns_center**2, axis=1)))
        ns_center = ns_center / m
        ns_center = torch.from_numpy(ns_center).float().unsqueeze(0).cuda() 

        # feature dynamics prediction
        normalized_action = normalized_action.float().unsqueeze(0).cuda()
        pred_features = self.feature_dynamics_network(z_states, ns_center, normalized_action)

        # decode the prediction
        ret_recon_next = self.dvae.decode_features(pred_features, neighborhood, ns_center, logits, normalized_state)
        recon_pcl = ret_recon_next[1]
        ns_pred = recon_pcl.squeeze().detach().cpu().numpy()

        # unnormalize the predicted next state
        ns_pred = ns_pred * m
        ns_pred = ns_pred + normalization_centroid
        return ns_pred
    
    def sample_action(self, mus,sigmas):
        '''
        Here we sample the actions from the normal distributions
        '''
        action = torch.normal(mus,sigmas)
        # x,y,z,rz,d = torch.normal(mus,sigmas)
        # action = [x,y,z,rz,d]
        action = action.detach().cpu().numpy()
        return action

    def plan(self, obs_pcl, target_pcl):
        """
        Use the Cross-Entropy Method to iteratively plan a sequence of actions to reach the target point cloud.
        """
        
        n_iters = 2 # 25 # 100
        n_trials = 500 # 1000
        n_elites = 50 # 50
        horizon = self.planning_horizon

        mus = 0.5*torch.ones(horizon,5)
        sigmas = 0.1*torch.ones(horizon,5)

        # execute the planning
        # for _ in tqdm(range(horizon)):
        for _ in tqdm(range(n_iters)):
            recon, actions = [], []
            with torch.no_grad():
                
                a_pop = 0
                for _ in tqdm(range(n_trials)):

                    action = self.sample_action(mus,sigmas)
                    state = obs_pcl

                    # remove the actions that are outside of the alowed bounds (0,1)
                    if torch.any(torch.tensor(action) > 1).item() or torch.any(torch.tensor(action) < 0).item():
                        print("not saving action because it is invalid")
                        pass
                    else:
                        for i in range(horizon):
                            state = self.dynamics_model(state, action[i,:])

                        # append state to recon list
                        next_recon = state
                        a_pop+=1
                        actions.append(action)
                        recon.append(next_recon)
                    
                cd_target = torch.from_numpy(np.expand_dims(target_pcl, axis=0)).float().cuda()
                cd_final_state = torch.from_numpy(np.array(recon)).float().cuda()
                dists = [chamfer_distance(fs.unsqueeze(0), cd_target)[0].cpu().detach().numpy() for fs in cd_final_state]
                dists = torch.from_numpy(np.array(dists))
                
                print("\ndists: ", dists)

                elites_index = torch.argsort(dists)[:n_elites].tolist()
                print("Best dists: ", dists[elites_index])
                print("elites idx: ", len(elites_index))
                print("N elites: ", n_elites)

                # TODO: need to index actions with elites_index and stack them
                # shape (n_elites, horizon, 5)
                action_tensor = torch.Tensor(actions)
                print("action_tensor shape: ", action_tensor.shape)
                elites = action_tensor[elites_index]
                print("Elites shape: ", elites.shape)
                # elites = [torch.stack((torch.from_numpy(actions[i]))) for i in elites_index]
                # elites = torch.stack(elites)

                # based on the elites, update the mu and sigma
                mus = torch.mean(elites,dim=0)
                sigmas = torch.std(elites,dim=0)

        print("Mu's: ", mus)
        print("\nSigmas: ", sigmas)
        action = self.sample_action(mus,sigmas)   
        print("action shape: ", action.shape)

        # action = []
        # for a in action_tensor:
        #     action.append(np.array(a))
        # print("\nSampled Action: ", action)

        # given the sample, predict the final state and CD
        
        action_traj = []
        for i in range(action.shape[0]):
            final_state = obs_pcl
            action_traj.append(action[i,:])
            final_state = self.dynamics_model(final_state, action[i,:])
        best_planned_dist = chamfer_distance(torch.from_numpy(np.expand_dims(target_pcl, axis=0)).float().cuda(), torch.from_numpy(np.expand_dims(final_state, axis=0)).float().cuda())[0].cpu().detach().numpy()

        # visualize best final clay state overlayed with the target
        best_final_state = final_state
        recon_pcl = np.reshape(best_final_state, (2048,3))
        target = np.reshape(target_pcl, (2048,3))
        target_pcl, pcl = plot_target_and_state_clouds(recon_pcl, target)
        o3d.visualization.draw_geometries([pcl, target_pcl])

        # return the CD between the predicted final state and target pcl
        return action_traj, final_state, best_planned_dist
        