import torch.nn as nn
import torch
import numpy as np
import networkx as nx
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from pointnet2_ops import pointnet2_utils
from geometric_utils import *
from dynamics.dynamics_dataset import DemoActionDataset, GeometricDataset

path = '/home/alison/Clay_Data/Fully_Processed/Aug29_Human_Demos'
dataset = GeometricDataset(path, 'shell_scaled')

test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120, 6000]
for index in test_samples:
    state, next_state, action = dataset.__getitem__(index)

    # get state centroid
    state = torch.unsqueeze(state, 0).cuda()
    group_func = Group(num_group = 64, group_size = 32)
    _, centroids = group_func(state)
    centroids = centroids.squeeze().cpu().detach().numpy()
    state = centroids

    _, ns_centroids = group_func(torch.unsqueeze(next_state, 0).cuda())
    ns = ns_centroids.squeeze().cpu().detach().numpy()

    # state = state.detach().numpy()
    action = action.detach().numpy()

    # unnormalize the action
    action = unnormalize_a(action)

    # center the action at the origin of the point cloud
    # pcl_center = np.array([0.6, 0.0, 0.24]) # verified same pcl center that processed point clouds
    pcl_center = np.array([0.6, 0.0, 0.25])
    action[0:3] = action[0:3] - pcl_center
    action[0:3] = action[0:3] + np.array([0.005, -0.002, 0.0]) # observational correction
    # print("action centered: ", action)

    # scale the action (multiply x,y,z,d by 10)
    action_scaled = action * 10
    action_scaled[3] = action[3] # don't scale the rotation
    len = 10 * 0.08 # 8cm scaled  to point cloud scaling # TODO: figure out grasp width scaling issue
        
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

    # get the points in state pcl inside gripper 1
    g1_idx = points_inside_rectangle(g1_points, None, state)
    inlier_pts = state.copy()

    # pointcloud with points inside rectangle
    g1_inside = state[g1_idx,:]
    g1_inside_pcl = o3d.geometry.PointCloud()
    g1_inside_pcl.points = o3d.utility.Vector3dVector(g1_inside)
    g1_inside_colors = np.tile(np.array([1, 0, 0]), (g1_inside.shape[0],1))
    g1_inside_pcl.colors = o3d.utility.Vector3dVector(g1_inside_colors)

    # get the displacement vector for the gripper 1 base
    g1_dir_unit = dir_vec_from_points(end_pts[0], points[0])
    g1_displacement_vec = end_pts[0] - points[0]

    # apply the displacement vector to all the points in the state point cloud
    g1_diffs = np.tile(end_pts[0], (inlier_pts[g1_idx,:].shape[0],1)) - inlier_pts[g1_idx,:] 
    g1_diffs = np.linalg.norm(g1_diffs, axis=1)
    inlier_pts[g1_idx,:] = inlier_pts[g1_idx,:] -  np.tile(g1_diffs, (3,1)).T * np.tile(g1_dir_unit, (inlier_pts[g1_idx,:].shape[0],1))

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

    # get the points in state pcl inside gripper 1
    g2_idx = points_inside_rectangle(g2_points, None, state)

    # pointcloud with points inside rectangle
    g2_inside = state[g2_idx,:]
    g2_inside_pcl = o3d.geometry.PointCloud()
    g2_inside_pcl.points = o3d.utility.Vector3dVector(g2_inside)
    g2_inside_colors = np.tile(np.array([1, 0, 0]), (g2_inside.shape[0],1))
    g2_inside_pcl.colors = o3d.utility.Vector3dVector(g2_inside_colors)

    # get the displacement vector for the gripper 1 base
    g2_dir_unit = dir_vec_from_points(end_pts[1], points[1])
    g2_displacement_vec = end_pts[1] - points[1]

    # apply the displacement vector to all the points in the state point cloud
    g2_diffs = np.tile(end_pts[1], (inlier_pts[g2_idx,:].shape[0],1)) - inlier_pts[g2_idx,:] 
    g2_diffs = np.linalg.norm(g2_diffs, axis=1)
    
    inlier_pts[g2_idx,:] = inlier_pts[g2_idx,:] -  np.tile(g2_diffs, (3,1)).T * np.tile(g2_dir_unit, (inlier_pts[g2_idx,:].shape[0],1))
    inliers = o3d.geometry.PointCloud()
    inliers.points = o3d.utility.Vector3dVector(inlier_pts)
    inlier_colors = np.tile(np.array([1, 0, 0]), (inlier_pts.shape[0],1))
    inliers.colors = o3d.utility.Vector3dVector(inlier_colors)

    g2 = o3d.geometry.LineSet()
    g2.points = o3d.utility.Vector3dVector(g2_points)
    g2.lines = o3d.utility.Vector2iVector(g2_lines)
    g2.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g2_lines.shape[0],1)))

    # test plot the point cloud and action and next state
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(state)
    og_colors = np.tile(np.array([0, 0, 1]), (state.shape[0],1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

    ns_pcl = o3d.geometry.PointCloud()
    # ns = next_state.detach().numpy()
    ns_pcl.points = o3d.utility.Vector3dVector(ns)
    ns_colors = np.tile(np.array([0, 1, 0]), (ns.shape[0],1))
    ns_pcl.colors = o3d.utility.Vector3dVector(ns_colors)

    ctr_action = o3d.geometry.PointCloud()
    action_cloud = action_scaled[0:3].reshape(1,3)
    print("Action: ", action_cloud)
    ctr_action.points = o3d.utility.Vector3dVector(action_scaled[0:3].reshape(1,3))
    ctr_colors = np.tile(np.array([1, 0, 0]), (1,1))
    ctr_action.colors = o3d.utility.Vector3dVector(ctr_colors)
    o3d.visualization.draw_geometries([og_pcl, ns_pcl, ctr_action, line_set, g1, g2])
    o3d.visualization.draw_geometries([og_pcl, ns_pcl, ctr_action, line_set, g1, g2, g1_inside_pcl, g2_inside_pcl])

    # get the point indices and the combined displacements
    idxs = np.concatenate((g1_idx, g2_idx))
    print("\n\nidxs shape: ", idxs.shape)
    disp = np.concatenate((np.tile(g1_diffs, (3,1)).T * np.tile(g1_dir_unit, (inlier_pts[g1_idx,:].shape[0],1)), 
                           np.tile(g2_diffs, (3,1)).T * np.tile(g2_dir_unit, (inlier_pts[g2_idx,:].shape[0],1))))
    print("disp shape: ", disp.shape)

    G = create_graph_from_point_cloud(centroids, k=5)
    new_graph = propagate_displacements(G, idxs, disp, 0.6, 0.08)

    ns_ctroid = create_point_cloud_from_graph(new_graph)
    ns_pred = o3d.geometry.PointCloud()
    ns_pred.points = o3d.utility.Vector3dVector(ns_ctroid)
    ns_pred_colors = np.tile(np.array([0, 0, 0]), (ns_ctroid.shape[0],1))
    ns_pred.colors = o3d.utility.Vector3dVector(ns_pred_colors)



    o3d.visualization.draw_geometries([ns_pcl, ctr_action, line_set, g1, g2, inliers, ns_pred])
