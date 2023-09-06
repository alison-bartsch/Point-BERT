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
from chamferdist import ChamferDistance
from planners.geometric_sampler import GeometricSampler

class MPC():
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

    def plan(self, obs_pcl, target_pcl):
        """
        Use model predictive control to unroll the dynamics model and find the best action sequence
        given the current point cloud observation and a target point cloud.
        """
        final_states, all_actions = [], []
        state = obs_pcl
        for _ in tqdm(range(self.n_actions)):
            with torch.no_grad():

                actions = []
                for _ in range(self.planning_horizon):
                    if self.sampler == 'constrain_d':
                        a = random_sample_normalized_constrained_action(self.action_dim)
                    elif self.sampler == 'geometric_informed':
                        geometric = GeometricSampler(None)
                        a = geometric.geometric_sampler(state, target_pcl, n_samples=25)
                        a = normalize_action(a, self.action_dim)
                    else:
                        a = random_sample_normalized_action(self.action_dim)

                    # predict the next state given the proposed action
                    # print("\n\na: ", a)
                    state = self.dynamics_model(state, a)
                    # TODO: convert state to numpy array with 2D

                    # unnormalize action
                    unnorm_a = unnormalize_a(a)
                    actions.append(unnorm_a)
                final_states.append(state)
                all_actions.append(actions)
        
        cd_target = torch.from_numpy(np.expand_dims(target_pcl, axis=0)).float().cuda()
        cd_final_state = torch.from_numpy(np.array(final_states)).float().cuda()
        chamferDist = ChamferDistance()
        dists = [chamferDist(fs.unsqueeze(0), cd_target).detach().cpu().item() for fs in cd_final_state]
        # dists = [chamfer_distance(fs.unsqueeze(0), cd_target)[0].cpu().detach().numpy() for fs in cd_final_state]
        # print("dists: ", dists)

        # get the index of the smallest distance
        idx = np.argmin(dists)
        best_planned_dist = dists[idx]
        print("\nBest Planned Dist: ", best_planned_dist)

        # visualize best final clay state overlayed with the target
        best_final_state = final_states[idx]
        recon_pcl = np.reshape(best_final_state, (2048,3))
        target = np.reshape(target_pcl, (2048,3))
        target_pcl, pcl = plot_target_and_state_clouds(recon_pcl, target)
        o3d.visualization.draw_geometries([pcl, target_pcl])

        final_state = final_states[-1]
        return all_actions[idx], final_state, best_planned_dist

        