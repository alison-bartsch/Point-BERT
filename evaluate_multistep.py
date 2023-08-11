import torch
import numpy as np
import open3d as o3d
from models.dvae import *
from tqdm import tqdm
from tools import builder
from utils.config import cfg_from_yaml_file
from dynamics.dynamics_dataset import DemoActionDataset, DemoWordDataset

from torch_geometric.data import Data, Batch
from torch_cluster import knn_graph
from pytorch3d.loss import chamfer_distance


# experiment name and details
exp_name = None

# load all necessary models
path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'
dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
device = torch.device('cuda')

# # for the models that were trained separately
# center_dynamics_path = 'centroid_experiments/exp1' # 'dvae_dynamics_experiments/exp16_center_pointnet'
# feature_dynamics_path = 'dvae_dynamics_experiments/exp39_dgcnn_pointnet'
# checkpoint = torch.load(feature_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
# feature_dynamics_network = checkpoint['dynamics_network'].to(device)
# ctr_checkpoint = torch.load(center_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
# center_dynamics_network = ctr_checkpoint['center_dynamics_network'].to(device)

# # for the models that were trained together
# dynamics_path = 'august_experiments/exp6'
# checkpoint = torch.load(dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
# center_dynamics_network = checkpoint['center_dynamics_network'].to(device)
# feature_dynamics_network = checkpoint['feature_dynamics_network'].to(device)

center_dynamics_path = 'centroid_experiments/exp3'
ctr_checkpoint = torch.load(center_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
center_dynamics_network = ctr_checkpoint['center_dynamics_network'].to(device)

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
n_steps = 5
for index in tqdm(range(9419 - 60*(n_steps - 1))): # 9419 - 120
# test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120]
# for index in test_samples:
    for i in range(n_steps): # 3-step dynamics evaluation
        item_index = index + i*60

        state, next_state, action = dataset.__getitem__(item_index)
        state = torch.unsqueeze(state, 0)
        next_state = torch.unsqueeze(next_state, 0)
        action = torch.unsqueeze(action, 0)

        # # create og full point cloud [BLUE]
        # ns = next_state.squeeze().detach().cpu().numpy()
        # og_pcl = o3d.geometry.PointCloud()
        # og_pcl.points = o3d.utility.Vector3dVector(ns)
        # og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
        # og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

        state = state.cuda()
        next_state = next_state.cuda()
        action = action.cuda()

        if i == 0:
            delta_ns = center_dynamics_network(state, action).to(device)
        else:
            delta_ns = center_dynamics_network(pred_ns, action).to(device)
        pred_ns = state + delta_ns

    # # visualize reconstructed full cloud cloud [RED]
    # recon_pcl = pred_ns.squeeze().detach().cpu().numpy()
    # recon_pcl = np.reshape(recon_pcl, (1048, 3))
    # pcl = o3d.geometry.PointCloud()
    # pcl.points = o3d.utility.Vector3dVector(recon_pcl)
    # pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
    # pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
    # o3d.visualization.draw_geometries([pcl, og_pcl]) # , vae_pcl])

    # get chamfer distance between recon_pcl and ns 
    chamfer_full_cloud = chamfer_distance(next_state, pred_ns)[0].cpu().detach().numpy()
    cds_full_cloud.append(chamfer_full_cloud)

mean_cd = np.mean(cds_full_cloud)
print("Mean CD: ", mean_cd)
# mean_centroid_cd = np.mean(cds_centroid)