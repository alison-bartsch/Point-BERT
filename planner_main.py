import os
import time
import ast
import torch
import json
import random
import argparse
import numpy as np
import cv2
import queue
import threading
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
from planners.cem import CEM
import planners.UdpComms as U
from chamferdist import ChamferDistance

# import sys
# sys.path.append("./MSN-Point-Cloud-Completion/emd/")
# import emd_module as emd


# start by naming the experiment and the experiment parameters
# create a folder with the experiment name and save a json file with the experiment parameters
# create two threads: one for the external camera to collect a video and one for the planning / communitation with robot computer

# VIDEO THREAD
def video_loop(cam_pipeline, save_path, done_queue):
    forcc = cv2.VideoWriter_fourcc(*'XVID')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    out = cv2.VideoWriter(save_path + '/video.avi', forcc, 30.0, (848, 480))

    # record until main loop is complete
    while done_queue.empty():
        frames = cam_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        out.write(color_image)
    
    cam_pipeline.stop()
    out.release()

# MAIN THREAD:
def main_loop(cam_pipelines, cam_streams, udp, target_pcl, exp_args, save_path, dvae, feature_dynamics_network, done_queue):
    """
    """

    # initialize planner
    if exp_args['mpc'] == True:
        device = torch.device('cuda')
        planner = MPC(device, dvae, feature_dynamics_network, exp_args['action_horizon'], exp_args['n_actions'], exp_args['a_dim'], sampler=exp_args['sampler'])
    elif exp_args['cem'] == True:
        device = torch.device('cuda')
        planner = CEM(device, dvae, feature_dynamics_network, exp_args['action_horizon'], exp_args['n_actions'], exp_args['a_dim'], sampler='random')
  
    cur_CDs = []
    best_planned_dists = []

    for i in range(25):
        # get point clouds from each camera
        pc2 = get_camera_point_cloud(cam_pipelines[2], cam_streams[2])
        pc3 = get_camera_point_cloud(cam_pipelines[3], cam_streams[3])
        pc4 = get_camera_point_cloud(cam_pipelines[4], cam_streams[4])
        pc5 = get_camera_point_cloud(cam_pipelines[5], cam_streams[5])

        # process the clouds
        obs = fuse_point_clouds(pc2, pc3, pc4, pc5)

        # # FOR TESTING
        # obs = np.load( '/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos/States/shell_scaled_state0.npy')

        print("\nMin of obs: ", np.min(obs, axis=0))
        print("Max of obs: ", np.max(obs, axis=0))
        print("Min of target: ", np.min(target_pcl, axis=0))
        print("Max of target: ", np.max(target_pcl, axis=0))

        np.save(save_path + '/obs' + str(i*exp_args['n_replan']) + '.npy', obs)

        # print distance between target and current state
        eval_target = torch.from_numpy(np.expand_dims(target_pcl, axis=0)).cuda()
        eval_obs = torch.from_numpy(np.expand_dims(obs, axis=0)).cuda()
        dist = chamfer_distance(eval_target, eval_obs)[0].cpu().detach().numpy()
        print("\nCurrent Chamfer Distance: ", dist)

        # EMD = emd.emdModule()
        # dist, _ = EMD(eval_target, eval_obs, eps=0.005, iters=50)
        # dist = torch.sqrt(dist).mean(1).cpu().detach().numpy()[0]
        # cur_CDs.append(dist)
        # print("\nCurrent EMD Distance: ", dist)

        # save the current state and target point cloud
        np.save(save_path + '/cur_state_pcl' + str(i*exp_args['n_replan']) + '.npy', obs)
        np.save(save_path + '/target_pcl' + str(i*exp_args['n_replan']) + '.npy', target_pcl)

        target_pcl_plot, cur_state_pcl = plot_target_and_state_clouds(obs, target_pcl)
        pcl_to_image(cur_state_pcl, target_pcl_plot, save_path + '/cloud_states_' + str(i*exp_args['n_replan']) + '.png')

        # save camera image views
        save_rgb_image(cam_pipelines[2], cam_streams[2], save_path + '/img_cam2_state' + str(i*exp_args['n_replan']) + '.jpg')
        save_rgb_image(cam_pipelines[3], cam_streams[3], save_path + '/img_cam3_state' + str(i*exp_args['n_replan']) + '.jpg')
        save_rgb_image(cam_pipelines[4], cam_streams[4], save_path + '/img_cam4_state' + str(i*exp_args['n_replan']) + '.jpg')
        save_rgb_image(cam_pipelines[5], cam_streams[5], save_path + '/img_cam5_state' + str(i*exp_args['n_replan']) + '.jpg')

        # generate plan
        start = time.time()
        action_plan, final_state, best_planned_dist = planner.plan(np.asarray(cur_state_pcl.points), np.asarray(target_pcl_plot.points))
        end = time.time()
        print("\n\n\nTotal Planning Time: ", end-start)
        best_planned_dists.append(best_planned_dist)

        # check to make sure the best planned distance is less than the current distance
        if best_planned_dist >= dist + 0.0001:
            print("\nPlanned actions not improving CD, stopping planning!")
            break

        # save the predicted final state of selected trajectory along with the action plan (unnormalized actions) and CD
        np.save(save_path + '/action_plan' + str(i*exp_args['n_replan']) + '.npy', action_plan)
        np.save(save_path + '/final_state' + str(i*exp_args['n_replan']) + '.npy', final_state)
        np.save(save_path + '/best_planned_chamfer_dist' + str(i*exp_args['n_replan']) + '.npy', best_planned_dist)
        
        print("Action Plan: ", action_plan)

        # create list of lists of actions
        lst = [list(x) for x in action_plan]
        print("Lst: ", lst)
        
        message = str(lst)

        print("converted: ", ast.literal_eval(message))

        # SEND ACTIONS TO CONTROL TO EXECUTE
        print("\n\nMessage Sent:\n", message)
        # print("Centroid list: ", centroid_list)
        udp.SendData(str(message))  # Send the message

        # WAIT FOR DONE SIGNAL FROM CONTROL
        while True:
            replan_msg = udp.ReadReceivedData()
            if replan_msg is not None:
                print("action: ", replan_msg)
                break
    
    done_queue.put("Done!")

if __name__=='__main__':
    # import the dvae and dynamics models
    dvae_path = 'Point-BERT/experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
    device = torch.device('cuda')

    # for the models that were trained separately
    feature_dynamics_path = 'Point-BERT/feature_experiments/exp15_new_dataset' # 'dvae_dynamics_experiments/exp39_dgcnn_pointnet'
    checkpoint = torch.load(feature_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
    feature_dynamics_network = checkpoint['feature_dynamics_network'].to(device) # 'dynamics_network'

    # load the dvae model
    config = cfg_from_yaml_file('Point-BERT/cfgs/Dynamics/dvae.yaml')
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
        'n_actions': 2500, # 1500 geometric = 25
        'a_dim': 5,
        'exp_name': 'Exp8',
        'sampler': 'random', # 'geometric_informed', 'random'
        'target_shape': 'X' # ['cone', 'cylinder', 'line', 'square', 'T', 'triangle', 'U', 'wavy', 'X']
    }

    # define target_pcl
    # X: 360
    # square: 1500
    # target_pcl = np.load('/home/alison/Documents/GitHub/Point-BERT/planners/Target_Shapes/' + exp_args['target_shape'] + '/state.npy')
    target_pcl = np.load('/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos/Next_States/shell_scaled_state360.npy')

    # define experiment name and save path
    # exp_name = 'Testing'
    save_path = join(os.getcwd(), 'Point-BERT/planners/Experiments/' + exp_args['exp_name'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # save json file with experiment parameters
    with open(save_path + '/exp_args.json', 'w') as fp:
        json.dump(exp_args, fp)

    # initialize UDP communication with robot computer
    # TODO: CHANGE TEH IP ADDRESS!!!!!!!
    # udp = U.UdpComms(udpIP='172.26.69.200', sendIP='172.26.5.54', portTX=5500, portRX=5501, enableRX=True)
    udp = U.UdpComms(udpIP='172.26.69.200', sendIP='172.26.229.213', portTX=5500, portRX=5501, enableRX=True)

    # initialize the cameras
    W = 848
    H = 480

    # ----- Camera 2 (static) -----
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device('151322066099')
    config_2.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config_2.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # ----- Camera 3 (static) -----
    pipeline_3 = rs.pipeline()
    config_3 = rs.config()
    config_3.enable_device('151322069488')
    config_3.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config_3.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # ----- Camera 4 (static) -----
    pipeline_4 = rs.pipeline()
    config_4 = rs.config()
    config_4.enable_device('151322061880')
    config_4.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config_4.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # ----- Camera 5 (static) -----
    pipeline_5 = rs.pipeline()
    config_5 = rs.config()
    config_5.enable_device('151322066932')
    config_5.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config_5.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # start streaming
    pipeline_2.start(config_2)
    pipeline_3.start(config_3)
    pipeline_4.start(config_4)
    pipeline_5.start(config_5)

    # align stream
    aligned_stream_2 = rs.align(rs.stream.color)
    aligned_stream_3 = rs.align(rs.stream.color)
    aligned_stream_4 = rs.align(rs.stream.color)
    aligned_stream_5 = rs.align(rs.stream.color)

    # point clouds
    point_cloud_2 = rs.pointcloud()
    point_cloud_3 = rs.pointcloud()
    point_cloud_4 = rs.pointcloud()
    point_cloud_5 = rs.pointcloud()

    # populate cam_pipelines dict
    cam_pipelines = {
        2: pipeline_2,
        3: pipeline_3,
        4: pipeline_4,
        5: pipeline_5
    }

    # populate the cam_streams dict
    cam_streams = {
        2: aligned_stream_2,
        3: aligned_stream_3,
        4: aligned_stream_4,
        5: aligned_stream_5
    }

    extra_cam_pipeline = rs.pipeline()
    extra_config = rs.config()
    extra_config.enable_device('152522250441')
    extra_config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    extra_cam_pipeline.start(extra_config)

    done_queue = queue.Queue()

    main_thread = threading.Thread(target=main_loop, args=(cam_pipelines, cam_streams, udp, target_pcl, exp_args, save_path, dvae, feature_dynamics_network, done_queue))
    video_thread = threading.Thread(target=video_loop, args=(extra_cam_pipeline, save_path, done_queue))

    main_thread.start()
    video_thread.start()

    # main_loop(cam_pipelines, cam_streams, udp, target_pcl, exp_args, save_path, dvae, feature_dynamics_network)