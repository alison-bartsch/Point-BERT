import torch
import numpy as np
import open3d as o3d
from models.dvae import *
from tqdm import tqdm
from tools import builder
from geometric_utils import *
from utils.config import cfg_from_yaml_file
from dynamics.dynamics_dataset import DemoActionDataset, GeometricDataset

from torch_geometric.data import Data, Batch
from torch_cluster import knn_graph
from pytorch3d.loss import chamfer_distance

import sys
sys.path.append("./MSN-Point-Cloud-Completion/emd/")
import emd_module as emd

idx1 = [0, 1043, 2034, 601, 459, 222, 120, 180, 240, 300, 360]
idx2 = [0, 900, 2022, 654, 987, 333, 180, 240, 300, 360, 420]

for i in range(len(idx1)):
    pc1 = np.load('/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos/States/shell_scaled_state' + str(idx1[i]) + '.npy')
    pc2 = np.load('/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos/States/shell_scaled_state' + str(idx2[i]) + '.npy')

    pc1_tensor = torch.from_numpy(pc1).unsqueeze(0).cuda()
    pc2_tensor = torch.from_numpy(pc2).unsqueeze(0).cuda()
    chamfer = chamfer_distance(pc1_tensor, pc2_tensor)[0].cpu().detach().numpy()

    EMD = emd.emdModule()
    dist, _ = EMD(pc1_tensor, pc2_tensor, eps=0.005, iters=50)
    emds = torch.sqrt(dist).mean(1).cpu().detach().numpy()[0]

    print("\nChamfer Distance: ", chamfer)
    print("EMD: ", emds)

    pcl1 = o3d.geometry.PointCloud()
    pcl1.points = o3d.utility.Vector3dVector(pc1)
    pcl1_colors = np.tile(np.array([1, 0, 0]), (len(pc1),1))
    pcl1.colors = o3d.utility.Vector3dVector(pcl1_colors)

    pcl2 = o3d.geometry.PointCloud()
    pcl2.points = o3d.utility.Vector3dVector(pc2)
    pcl2_colors = np.tile(np.array([0, 0, 1]), (len(pc2),1))
    pcl2.colors = o3d.utility.Vector3dVector(pcl2_colors)

    o3d.visualization.draw_geometries([pcl1, pcl2]) # , vae_pcl])