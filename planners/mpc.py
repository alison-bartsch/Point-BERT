import os
import time
import torch
import json
import random
import argparse
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from planners.robot_utils import *
from planners.plan_geometric_utils import * 
from os.path import join
from pytorch3d.loss import chamfer_distance
from chamferdist import ChamferDistance
from planners.geometric_sampler import *

# import sys
# sys.path.append("./MSN-Point-Cloud-Completion/emd/")
# import emd_module as emd

# mp.set_start_method('spawn')

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
    
    def dynamics_model_parallel(self, states, actions):
        """
        Assumes list of states and actions to get dynamics of
        """
        # print(len(states))
        # print(len(actions))
        # states = np.array(states)
        actions = np.array(actions)
        # print("States: ", states.shape)
        # print("Actions: ", actions.shape)

        normalized_state = torch.from_numpy(self.normalize_pcl(states)).float().cuda()
        normalized_action = torch.from_numpy(actions).cuda()
        # print("State shape: ", normalized_state.shape)
        # print("Action shape: ", normalized_action.shape)

        z_states, neighborhood, center, logits = self.dvae.encode(normalized_state) #.to(device)
        
        state = torch.from_numpy(states)
        action = torch.from_numpy(actions)

        state_torch = state.cuda()
        group_func = Group(num_group = 64, group_size = 32)
        _, centroids = group_func(state_torch)
        centroids = centroids.squeeze().cpu().detach().numpy()

        # convert state and action to list of numpy arrays
        state_list = centroids.tolist()
        action_list = actions.tolist()
        # print("State list shape: ", len(state))
        # ns_center_unnormalized = predict_centroid_dynamics(state, action)
        # pool = Pool(10)
        # ns_center_unnormalized_list = pool.map(self.multi_run_wrapper, zip(state_list, action_list))

        
        # print(mp.get_start_method())
        # start = time.time()
        ctx = mp.get_context('spawn')
        pool = mp.pool.Pool(context=ctx, processes=5) # 10
        # middle = time.time()
        ns_center_unnormalized_list = pool.map(self.multi_run_wrapper, zip(state_list, action_list))
        # end_time = time.time()
        # print("End time: ", end_time - middle)
        # print("Complete Time: ", end_time - start)
        # print("Middle Time: ", middle - start)

        # q = mp.Queue()
        # p = mp.Process(target=predict_centroid_dynamics_multiprocessing, args=(zip(state_list, action_list), q))
        # p.start()
        # p.join()
        # ns_center_unnormalized_list = p.get()

        # print("Len: ", len(ns_center_unnormalized_list))
        ns_center_unnormalized = np.array(ns_center_unnormalized_list)

        # print("Mean shape: ", np.mean(states, axis=1).shape)
        # print("Expanded shape: ", np.expand_dims(np.mean(states, axis=1), axis=1).shape)
        # print("Tile Shape: ", np.tile(np.expand_dims(np.mean(states, axis=1), axis=1), (1, 64, 1)).shape)
        # print("ns_center_unnormalized shape: ", ns_center_unnormalized.shape)

        normalization_centroid = np.tile(np.expand_dims(np.mean(states, axis=1), axis=1), (1, 64, 1))
        ns_center = ns_center_unnormalized - normalization_centroid
        m = np.max(np.sqrt(np.sum(ns_center**2, axis=1)))
        ns_center = ns_center / m
        ns_center = torch.from_numpy(ns_center).float().cuda() 

        # feature dynamics prediction
        normalized_action = normalized_action.float().cuda()
        # print("z_states: ", z_states.shape)
        # print("ns_center: ", ns_center.shape)
        # print("normalized_action: ", normalized_action.shape)
        pred_features = self.feature_dynamics_network(z_states, ns_center, normalized_action)

        # decode the prediction
        ret_recon_next = self.dvae.decode_features(pred_features, neighborhood, ns_center, logits, normalized_state)
        recon_pcl = ret_recon_next[1]
        ns_pred = recon_pcl.squeeze().detach().cpu().numpy()

        # unnormalize the predicted next state
        ns_pred = ns_pred * m
        normalization_state = np.tile(np.expand_dims(np.mean(states, axis=1), axis=1), (1, 2048, 1))
        ns_pred = ns_pred + normalization_state # normalization_centroid
        # print("ns pred shape: ", ns_pred.shape)

        # convert ns_pred from array to list of arrays
        ns_pred_list = ns_pred.tolist()
        # print("len ns pred: ", len(ns_pred_list))
        return ns_pred_list

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
    
    def visualize_grasp(self, state, action, next_state):
        """
        Visualize the grasp mesh on the point cloud showing the state and predicted next state.
        Assume the inputs are unnormalized point clouds and unnormalized action.
        """

        # visualize state cloud
        s_pcl = o3d.geometry.PointCloud()
        s_pcl.points = o3d.utility.Vector3dVector(state)
        s_colors = np.tile(np.array([0, 0, 1]), (state.shape[0],1))
        s_pcl.colors = o3d.utility.Vector3dVector(s_colors)

        # visualize reconstructed full cloud cloud [RED]
        ns_pcl = o3d.geometry.PointCloud()
        ns_pcl.points = o3d.utility.Vector3dVector(next_state)
        ns_pcl_colors = np.tile(np.array([1, 0, 0]), (next_state.shape[0],1))
        ns_pcl.colors = o3d.utility.Vector3dVector(ns_pcl_colors)

        # center the action at the origin of the point cloud
        pcl_center = np.array([0.6, 0.0, 0.25])
        action[0:3] = action[0:3] - pcl_center
        action[0:3] = action[0:3] + np.array([0.005, -0.002, 0.0]) # observational correction

        # scale the action (multiply x,y,z,d by 10)
        action_scaled = action * 10
        action_scaled[3] = action[3] # don't scale the rotation
        len = 10 * 0.08 # 8cm scaled  

        # get the points and lines for the action orientation visualization
        ctr = action_scaled[0:3]
        upper_ctr = ctr + np.array([0,0, 0.6])
        rz = 90 + action_scaled[3]
        points, lines = line_3d_start_end(ctr, rz, len)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (lines.shape[0],1)))

        delta = 0.8 - action_scaled[4] 
        end_pts, _ = line_3d_start_end(ctr, rz, len - delta)
        top_end_pts, _ = line_3d_start_end(upper_ctr, rz, len - delta)

        # get the top points for the grasp (given gripper finger height)
        top_points, _ = line_3d_start_end(upper_ctr, rz, len)

        # gripper 1 
        g1_base_start, _ = line_3d_start_end(points[0], rz+90, 0.18)
        g1_base_end, _ = line_3d_start_end(end_pts[0], rz+90, 0.18)
        g1_top_start, _ = line_3d_start_end(top_points[0], rz+90, 0.18)
        g1_top_end, _ = line_3d_start_end(top_end_pts[0], rz+90, 0.18)
        g1_points, g1_lines = line_3d_point_set([g1_base_start, g1_base_end, g1_top_start, g1_top_end])

        g1 = o3d.geometry.LineSet()
        g1.points = o3d.utility.Vector3dVector(g1_points)
        g1.lines = o3d.utility.Vector2iVector(g1_lines)
        g1.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g1_lines.shape[0],1)))

        # gripper 2
        g2_base_start, _ = line_3d_start_end(points[1], rz+90, 0.18)
        g2_base_end, _ = line_3d_start_end(end_pts[1], rz+90, 0.18)
        g2_top_start, _ = line_3d_start_end(top_points[1], rz+90, 0.18)
        g2_top_end, _ = line_3d_start_end(top_end_pts[1], rz+90, 0.18)
        g2_points, g2_lines = line_3d_point_set([g2_base_start, g2_base_end, g2_top_start, g2_top_end])

        g2 = o3d.geometry.LineSet()
        g2.points = o3d.utility.Vector3dVector(g2_points)
        g2.lines = o3d.utility.Vector2iVector(g2_lines)
        g2.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g2_lines.shape[0],1)))

        ctr_action = o3d.geometry.PointCloud()
        action_cloud = action_scaled[0:3].reshape(1,3)
        # print("Action: ", action_cloud)
        ctr_action.points = o3d.utility.Vector3dVector(action_scaled[0:3].reshape(1,3))
        ctr_colors = np.tile(np.array([1, 0, 0]), (1,1))
        ctr_action.colors = o3d.utility.Vector3dVector(ctr_colors)
        o3d.visualization.draw_geometries([s_pcl, ns_pcl, ctr_action, line_set, g1, g2])

    def multi_run_wrapper(self, args):
        return predict_centroid_dynamics_multiprocessing(*args)
        # return predict_centroid_dynamics_multiprocessing(*args)

    def parallel_plan(self, obs_pcl, target_pcl):
        n_parallel = 5 # 10
        final_states, all_actions = [], []
        for _ in tqdm(range(int(self.n_actions / n_parallel))):
            states = np.tile(obs_pcl, (n_parallel, 1, 1)) # TODO: may need to be a list

            a_list = []
            unnorm_a_list = []
            for _ in range(n_parallel):
                if self.sampler == 'constrain_d':
                    a = random_sample_normalized_constrained_action(self.action_dim)
                elif self.sampler == 'geometric_informed':
                    a = geometric_sample(states, target_pcl)
                    a = normalize_action(a, self.action_dim)
                else:
                    a = random_sample_normalized_action(self.action_dim)
                a_list.append(a)
                unnorm_a_list.append(unnormalize_a(a))

            # predict the next state given the proposed action
            next_states = self.dynamics_model_parallel(states, a_list)
            
            # TODO: add next_states list and unnorm_a_list to final_states and all_actions
            final_states = final_states + next_states
            all_actions = all_actions + unnorm_a_list
            
        cd_target = torch.from_numpy(np.expand_dims(target_pcl, axis=0)).float().cuda()
        cd_final_state = torch.from_numpy(np.array(final_states)).float().cuda()

        # EMD = emd.emdModule()
        # dists = [torch.sqrt(EMD(fs.unsqueeze(0), cd_target, eps=0.005, iters=50)[0]).mean(1).cpu().detach().numpy()[0] for fs in tqdm(cd_final_state)]
        
        dists = [chamfer_distance(fs.unsqueeze(0), cd_target)[0].cpu().detach().numpy() for fs in cd_final_state]
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

        # calculate min/max of point clouds for sanity check
        print("\nMin of target: ", np.min(target, axis=0))
        print("Max of target: ", np.max(target, axis=0))
        print("Min of recon_pcl: ", np.min(recon_pcl, axis=0))
        print("Max of recon_pcl: ", np.max(recon_pcl, axis=0))

        final_state = final_states[-1]
        return all_actions[idx], final_state, best_planned_dist


    def plan(self, obs_pcl, target_pcl):
        """
        Use model predictive control to unroll the dynamics model and find the best action sequence
        given the current point cloud observation and a target point cloud.
        """
        final_states, all_actions = [], []
        # state = obs_pcl
        for _ in tqdm(range(self.n_actions)):
            state = np.copy(obs_pcl) # condisder replacing with deepcopy
            with torch.no_grad():

                actions = []
                for _ in range(self.planning_horizon):
                    if self.sampler == 'constrain_d':
                        a = random_sample_normalized_constrained_action(self.action_dim)
                    elif self.sampler == 'geometric_informed':
                        # geometric = GeometricSampler(None)
                        # print("\nState min: ", np.min(state, axis=0))
                        a = geometric_sample(state, target_pcl)
                        # print("geometric action: ", a)
                        a = normalize_action(a, self.action_dim)
                        # print("normalized a :", a)
                    else:
                        a = random_sample_normalized_action(self.action_dim)

                    # predict the next state given the proposed action
                    state = self.dynamics_model(state, a)

                    # unnormalize action
                    unnorm_a = unnormalize_a(a)

                    # test visualize the grasp region for sanity check
                    # self.visualize_grasp(state, unnorm_a, obs_pcl)

                    actions.append(unnorm_a)
                final_states.append(state)
                all_actions.append(actions)
        
        cd_target = torch.from_numpy(np.expand_dims(target_pcl, axis=0)).float().cuda()
        cd_final_state = torch.from_numpy(np.array(final_states)).float().cuda()

        # EMD = emd.emdModule()
        # dists = [torch.sqrt(EMD(fs.unsqueeze(0), cd_target, eps=0.005, iters=50)[0]).mean(1).cpu().detach().numpy()[0] for fs in tqdm(cd_final_state)]
        
        dists = [chamfer_distance(fs.unsqueeze(0), cd_target)[0].cpu().detach().numpy() for fs in cd_final_state]
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

        # calculate min/max of point clouds for sanity check
        print("\nMin of target: ", np.min(target, axis=0))
        print("Max of target: ", np.max(target, axis=0))
        print("Min of recon_pcl: ", np.min(recon_pcl, axis=0))
        print("Max of recon_pcl: ", np.max(recon_pcl, axis=0))

        final_state = final_states[-1]
        return all_actions[idx], final_state, best_planned_dist

        