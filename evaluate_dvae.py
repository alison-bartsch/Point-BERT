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


# path = '/home/alison/Clay_Data/Fully_Processed/Aug24_Human_Demos_Fully_Processed'
path = 'home/alison/Clay_Data/Fully_Processed/dvae_test'
dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
device = torch.device('cuda')

# load the dvae model
config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
config=config.config
dvae = builder.model_builder(config)
builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
dvae.to(device)
dvae.eval()

# initialize the dataset
dataset = DemoActionDataset(path, 'shell_scaled')

test_samples = [0,1,2,3]
for index in test_samples:
    state, next_state, action = dataset.__getitem__(index)
    state = torch.unsqueeze(state, 0)

    # create og full point cloud [BLUE]
    s = state.squeeze().detach().cpu().numpy()
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(s)
    og_colors = np.tile(np.array([0, 0, 1]), (len(s),1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)
    o3d.visualization.draw_geometries([og_pcl])

    state = state.cuda()

    state_vae_decoded = dvae.forward(state)

    # z_states, neighborhood, center, logits = dvae.encode(state) 
    # state_vae_decoded = dvae.decode(z_states, neighborhood, center, logits, state)

    # create vae recon full point cloud [RED]
    vae_ns = state_vae_decoded[1].squeeze().detach().cpu().numpy()
    vae_pcl = o3d.geometry.PointCloud()
    vae_pcl.points = o3d.utility.Vector3dVector(vae_ns)
    vae_colors = np.tile(np.array([1, 0, 0]), (len(vae_ns),1))
    vae_pcl.colors = o3d.utility.Vector3dVector(vae_colors)
    o3d.visualization.draw_geometries([vae_pcl])

    chamfer_full_cloud = chamfer_distance(state, state_vae_decoded[1])[0].cpu().detach().numpy()
    print("CD: ", chamfer_full_cloud)
