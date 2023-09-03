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
from os.path import join
from planners.geometric_sampler import GeometricSampler
from planners.robot_utils import *
from planners.plan_geometric_utils import *
from pytorch3d.loss import chamfer_distance
from utils.config import cfg_from_yaml_file
from models.dvae import *
from tools import builder
from planners.mpc import MPC
import planners.UdpComms as U


# start by naming the experiment and the experiment parameters
# create a folder with the experiment name and save a json file with the experiment parameters
# create two threads: one for the external camera to collect a video and one for the planning / communitation with robot computer

# MAIN THREAD:
def main(cam_pipelines, cam_streams, udp, target_pcl, exp_args, save_path, dvae, feature_dynamics_network):
    """
    """

    # initialize planner
    if exp_args['mpc'] == True:
        device = torch.device('cuda')
        planner = MPC(device, dvae, feature_dynamics_network, exp_args['action_horizon'], exp_args['n_actions'], exp_args['a_dim'], sampler='random')
    elif exp_args['cem'] == True:
        pass
        # TODO: initialize CEM planner
        # action_plan = CEM(encoder, decoder, forward_model, device, obs, target, action_horizon)
        
    cur_CDs = []
    decoded_dists = []
    best_planned_dists = []

    for i in range(int(exp_args['action_horizon'] / exp_args['n_replan'])):
        # # get point clouds from each camera
        # pc2 = get_camera_point_cloud(cam_pipelines[2], cam_streams[2])
        # pc3 = get_camera_point_cloud(cam_pipelines[3], cam_streams[3])
        # pc4 = get_camera_point_cloud(cam_pipelines[4], cam_streams[4])
        # pc5 = get_camera_point_cloud(cam_pipelines[5], cam_streams[5])

        # # process the clouds
        # obs = fuse_point_clouds(pc2, pc3, pc4, pc5)

        # FOR TESTING
        obs = np.load( '/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos/States/shell_scaled_state5.npy')

        np.save(save_path + '/obs' + str(i*exp_args['n_replan']) + '.npy', obs)

        # print distance between target and current state
        eval_target = torch.from_numpy(np.expand_dims(target_pcl, axis=0)).cuda()
        eval_obs = torch.from_numpy(np.expand_dims(obs, axis=0)).cuda()
        dist = chamfer_distance(eval_target, eval_obs)[0].cpu().detach().numpy()
        cur_CDs.append(dist)
        print("\nCurrent Chamfer Distance: ", dist)

        np.save(save_path + '/cur_state_pcl' + str(i*exp_args['n_replan']) + '.npy', obs)
        np.save(save_path + '/target_pcl' + str(i*exp_args['n_replan']) + '.npy', target_pcl)

        target_pcl, cur_state_pcl = plot_target_and_state_clouds(obs, target_pcl)
        pcl_to_image(cur_state_pcl, target_pcl, save_path + '/cloud_states_' + str(i*exp_args['n_replan']) + '.png')

        # # save camera image views
        # save_rgb_image(cam_pipelines[2], cam_streams[2], save_path + '/img_cam2_state' + str(i*n_replan) + '.jpg')
        # save_rgb_image(cam_pipelines[3], cam_streams[3], save_path + '/img_cam3_state' + str(i*n_replan) + '.jpg')
        # save_rgb_image(cam_pipelines[4], cam_streams[4], save_path + '/img_cam4_state' + str(i*n_replan) + '.jpg')
        # save_rgb_image(cam_pipelines[5], cam_streams[5], save_path + '/img_cam5_state' + str(i*n_replan) + '.jpg')

        # generate plan
        start = time.time()
        action_plan, decoded_dist, best_planned_dist = planner.plan(np.asarray(cur_state_pcl.points), np.asarray(target_pcl.points))
        end = time.time()
        print("\nTotal Planning Time: ", end-start)
        decoded_dists.append(decoded_dist)
        best_planned_dists.append(best_planned_dist)

        print("Action Plan: ", action_plan)

        # print("Unnormalized First Action: ", unnormalize_a(action_plan[0]))

        # # TODO: SEND ACTIONS TO CONTROL TO EXECUTE
        # print("\nMessage Sent:\n", message)
        # # print("Centroid list: ", centroid_list)
        # udp.SendData(str(message))  # Send the message

        # # TODO: WAIT FOR DONE SIGNAL FROM CONTROL
        # while True:
        #     replan_msg = udp.ReadReceivedData()
        #     if replan_msg is not None:
        #         target_list = replan_msg.split(',')
        #         print(target_list)
        #         break

if __name__=='__main__':
    # import the dvae and dynamics models
    dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
    device = torch.device('cuda')

    # for the models that were trained separately
    center_dynamics_path = 'centroid_experiments/exp33_geometric' # 'dvae_dynamics_experiments/exp16_center_pointnet'
    feature_dynamics_path = 'feature_experiments/exp15_new_dataset' # 'dvae_dynamics_experiments/exp39_dgcnn_pointnet'
    checkpoint = torch.load(feature_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
    feature_dynamics_network = checkpoint['feature_dynamics_network'].to(device) # 'dynamics_network'

    # load the dvae model
    config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
    config=config.config
    dvae = builder.model_builder(config)
    builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
    dvae.to(device)
    dvae.eval()

    # create the exp_args dictionary
    exp_args = {
        'action_horizon': 1, # 10
        'n_replan': 1,
        'mpc': True,
        'cem': False,
        'n_actions': 2, # 100
        'a_dim': 5
    }

    # define target_pcl
    target_pcl = np.load( '/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos/States/shell_scaled_state10.npy')

    # define experiment name and save path
    exp_name = 'Testing'
    save_path = join(os.getcwd(), 'planners/Experiments/' + exp_name)
    # os.mkdir(save_path)

    # initialize UDP communication with robot computer
    # TODO: CHANGE TEH IP ADDRESS!!!!!!!
    udp = U.UdpComms(udpIP='172.26.69.200', sendIP='172.26.5.54', portTX=5500, portRX=5501, enableRX=True)

    # initialize the cameras
    W = 848
    H = 480

    # # ----- Camera 2 (static) -----
    # pipeline_2 = rs.pipeline()
    # config_2 = rs.config()
    # config_2.enable_device('151322066099')
    # config_2.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    # config_2.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # # ----- Camera 3 (static) -----
    # pipeline_3 = rs.pipeline()
    # config_3 = rs.config()
    # config_3.enable_device('151322069488')
    # config_3.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    # config_3.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # # ----- Camera 4 (static) -----
    # pipeline_4 = rs.pipeline()
    # config_4 = rs.config()
    # config_4.enable_device('151322061880')
    # config_4.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    # config_4.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # # ----- Camera 5 (static) -----
    # pipeline_5 = rs.pipeline()
    # config_5 = rs.config()
    # config_5.enable_device('151322066932')
    # config_5.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    # config_5.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # # start streaming
    # pipeline_2.start(config_2)
    # pipeline_3.start(config_3)
    # pipeline_4.start(config_4)
    # pipeline_5.start(config_5)

    # # align stream
    # aligned_stream_2 = rs.align(rs.stream.color)
    # aligned_stream_3 = rs.align(rs.stream.color)
    # aligned_stream_4 = rs.align(rs.stream.color)
    # aligned_stream_5 = rs.align(rs.stream.color)

    # # point clouds
    # point_cloud_2 = rs.pointcloud()
    # point_cloud_3 = rs.pointcloud()
    # point_cloud_4 = rs.pointcloud()
    # point_cloud_5 = rs.pointcloud()

    # # populate cam_pipelines dict
    # cam_pipelines = {
    #     2: pipeline_2,
    #     3: pipeline_3,
    #     4: pipeline_4,
    #     5: pipeline_5
    # }

    # # populate the cam_streams dict
    # cam_streams = {
    #     2: aligned_stream_2,
    #     3: aligned_stream_3,
    #     4: aligned_stream_4,
    #     5: aligned_stream_5
    # }

    cam_pipelines = {}
    cam_streams = {}

    main(cam_pipelines, cam_streams, udp, target_pcl, exp_args, save_path, dvae, feature_dynamics_network)