import torch
import numpy as np
import open3d as o3d
from models.dvae import *
from tools import builder
from utils.config import cfg_from_yaml_file
from dynamics.dynamics_dataset import DemoActionDataset, DemoWordDataset

from torch_geometric.data import Data, Batch
from torch_cluster import knn_graph
from pytorch3d.loss import chamfer_distance


# load all the necessary models

# import the data

# create a folder to save the evaluation results

# iterate through the dataset
    # get the chamfer distance of the predictions
    # every N steps, save the gt and predictions of the centroids and next states
        # modify open3d to save with larger points (easier visibility)
        # save as a png

# report the chamfer distance

# experiment name and details
exp_name = None

# load all necessary models
# path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'

# path = '/home/alison/Clay_Data/Fully_Processed/May4_5D'
# path = '/home/alison/Clay_Data/Fully_Processed/Aug15_5D_Human_Demos'
# path = '/home/alison/Clay_Data/Fully_Processed/Aug24_Human_Demos_Fully_Processed'
path = '/home/alison/Clay_Data/Fully_Processed/Tiny_Dataset'
dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
device = torch.device('cuda')

# for the models that were trained separately
center_dynamics_path = 'centroid_experiments/exp31_tiny_dataset' # 'dvae_dynamics_experiments/exp16_center_pointnet'
feature_dynamics_path = 'feature_experiments/exp9_human_demos' # 'dvae_dynamics_experiments/exp39_dgcnn_pointnet'
checkpoint = torch.load(feature_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
feature_dynamics_network = checkpoint['feature_dynamics_network'].to(device) # 'dynamics_network'
ctr_checkpoint = torch.load(center_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
center_dynamics_network = ctr_checkpoint['center_dynamics_network'].to(device)

# # for the models that were trained together
# dynamics_path = 'august_experiments/exp6'
# checkpoint = torch.load(dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
# center_dynamics_network = checkpoint['center_dynamics_network'].to(device)
# feature_dynamics_network = checkpoint['feature_dynamics_network'].to(device)

# load the dvae model
config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
config=config.config
dvae = builder.model_builder(config)
builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
dvae.to(device)
dvae.eval()

# initialize the dataset
dataset = DemoActionDataset(path, 'shell_scaled')

# create a folder to save the evaluation results

# track chamfer distance
cds_full_cloud = []
cds_centroid = []

# iterate through the dataset
# for index in range(9419):
# test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3005, 3060, 3122, 6011, 7048]
test_samples = [0,1,2,3,4,5,6,7,8,9]
for index in test_samples:
    state, next_state, action = dataset.__getitem__(index)
    state = torch.unsqueeze(state, 0)
    next_state = torch.unsqueeze(next_state, 0)
    action = torch.unsqueeze(action, 0)

    # if index % 60 == 0:
        # save the combined clouds and the individual clouds as images with enlarged points
        # save their associated CD's 
    # elif cd < 0.009: # a particularly good prediction
    # elif cd > 0.015: # a particularly poor prediction

    # create og full point cloud [BLUE]
    ns = next_state.squeeze().detach().cpu().numpy()
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(ns)
    og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

    state = state.cuda()
    next_state = next_state.cuda()
    action = action.cuda()

    z_states, neighborhood, center, logits = dvae.encode(state) #.to(device)
    z_ns, ns_neighborhood, ns_gt_center, ns_logits = dvae.encode(next_state)
    ns_vae_decoded = dvae.decode(z_ns, ns_neighborhood, ns_gt_center, ns_logits, next_state)

    # print(center)
    # print(center.size)
    # print(center.size())

    # ns_center = center_dynamics_network(center, action).to(device)
    pred_ns_center = center_dynamics_network(center, action).to(device)
    # ns_center = center + pred_ns_center
    ns_center = pred_ns_center
    # ns_center = ns_gt_center 
    pred_features = feature_dynamics_network(z_states, ns_center, action)
    # pred_features = feature_dynamics_network(z_states, ns_gt_center, action)

    ret_recon_next = dvae.decode_features(pred_features, neighborhood, ns_center, logits, state)
    recon_pcl = ret_recon_next[1]

    # # create vae recon full point cloud [GREEN]
    # vae_ns = ns_vae_decoded[1].squeeze().detach().cpu().numpy()
    # vae_pcl = o3d.geometry.PointCloud()
    # vae_pcl.points = o3d.utility.Vector3dVector(vae_ns)
    # vae_colors = np.tile(np.array([0, 1, 0]), (len(vae_ns),1))
    # vae_pcl.colors = o3d.utility.Vector3dVector(vae_colors)

    # visualize reconstructed full cloud cloud [RED]
    recon_pcl = recon_pcl.squeeze().detach().cpu().numpy()
    recon_pcl = np.reshape(recon_pcl, (2048, 3))
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(recon_pcl)
    pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
    pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
    o3d.visualization.draw_geometries([pcl, og_pcl]) # , vae_pcl])

    # create og centroid point cloud [BLUE]
    # ns_center = ns_gt_center.squeeze().detach().cpu().numpy()
    ns_center_pcl = ns_gt_center.squeeze().cpu().numpy()
    og_pcl_cent = o3d.geometry.PointCloud()
    og_pcl_cent.points = o3d.utility.Vector3dVector(ns_center_pcl)
    og_colors_cent = np.tile(np.array([0, 0, 1]), (len(ns_center_pcl),1))
    og_pcl_cent.colors = o3d.utility.Vector3dVector(og_colors_cent)

    # create state centroid point cloud [GREEN]
    s_center_pcl = center.squeeze().cpu().numpy()
    s_pcl = o3d.geometry.PointCloud()
    s_pcl.points = o3d.utility.Vector3dVector(s_center_pcl)
    s_colors = np.tile(np.array([0, 1, 0]), (len(s_center_pcl),1))
    s_pcl.colors = o3d.utility.Vector3dVector(s_colors)
    
    # visualize reconstructed centoird point cloud [RED]
    pred_ns_center = ns_center.squeeze().detach().cpu().numpy()
    ns_pcl_cent = o3d.geometry.PointCloud()
    ns_pcl_cent.points = o3d.utility.Vector3dVector(pred_ns_center)
    ns_pcl_colors = np.tile(np.array([1, 0, 0]), (len(pred_ns_center),1))
    ns_pcl_cent.colors = o3d.utility.Vector3dVector(ns_pcl_colors)
    o3d.visualization.draw_geometries([s_pcl, og_pcl_cent, ns_pcl_cent])

    # get chamfer distance between recon_pcl and ns 
    chamfer_full_cloud = chamfer_distance(next_state, ret_recon_next[1])[0].cpu().detach().numpy()
    cds_full_cloud.append(chamfer_full_cloud)
    # print("CD: ", chamfer_full_cloud)
    # chamfer_centroids = chamfer_distance(ns_gt_center, ns_center)[0].cpu().detach().numpy()
    # cds_centroid.append(chamfer_centroids)

mean_cd = np.mean(cds_full_cloud)
print("Mean CD: ", mean_cd)
# mean_centroid_cd = np.mean(cds_centroid)