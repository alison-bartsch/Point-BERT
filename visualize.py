import torch
import numpy as np
import open3d as o3d
from models.dvae import *
from tools import builder
from utils.config import cfg_from_yaml_file
from dynamics.dynamics_dataset import DemoActionDataset

"""
Visualize the reconstructed predition of the dynamics model
"""
# define the action space and dynamics loss type
path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'
dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
dynamics_path = 'dvae_dynamics_experiments/exp1_twonetworks'
# actions = np.load(path + '/action_normalized.npy')

# load the dvae model
config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
config=config.config
dvae = builder.model_builder(config)
builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
device = torch.device('cuda')
dvae.to(device)

# load the checkpoint
checkpoint = torch.load(dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
dynamics_network = checkpoint['dynamics_network'].to(device)

# initialize the dataset
dataset = DemoActionDataset(path, 'shell_scaled')

# iterate through a few state/next state pairs
test_samples = [0, 6, 100, 3000, 5067, 2048, 2678, 3333, 6983, 222, 468, 172]
for index in test_samples:
    state, next_state, action = dataset.__getitem__(index)
    state = torch.unsqueeze(state, 0)
    next_state = torch.unsqueeze(next_state, 0)
    action = torch.unsqueeze(action, 0)

    # create og point cloud
    ns = next_state.squeeze().detach().cpu().numpy()
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(ns)
    og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)
    # o3d.visualization.draw_geometries([og_pcl])

    state = state.cuda()
    next_state = state.cuda()
    action = action.cuda()

    z_states, neighborhood, center, logits = dvae.encode(state) #.to(device)
    z_pred_next_states = dynamics_network(z_states, action).to(device)
    ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, state) #.to(device)
    recon_pcl = ret_recon_next[1]

    # visualize reconstructed cloud
    recon_pcl = recon_pcl.squeeze().detach().cpu().numpy()
    recon_pcl = np.reshape(recon_pcl, (2048, 3))
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(recon_pcl)
    pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
    pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
    # o3d.visualization.draw_geometries([pcl])
    o3d.visualization.draw_geometries([pcl, og_pcl])