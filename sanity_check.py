import torch
import numpy as np
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

# sanity check the dynamics model in planners/cem.py and planners/mpc.py
# along with the chamfer distance calculation to verify that the dynamics model is working


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
    'action_horizon': 4, # 10
    'n_replan': 1,
    'mpc': True,
    'cem': False,
    'n_actions': 1000, # 100
    'a_dim': 5,
    'target_shape': 'X'
}

path = '/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos'
idx = 480
# define target_pcl
gt_ns = np.load(path + '/Next_States/shell_scaled_state' + str(idx) + '.npy')
state = np.load(path + '/States/shell_scaled_state' + str(idx) + '.npy')

# import an action
actions = np.load(path + '/action_normalized.npy')
a = actions[idx]

# initialize planner
if exp_args['mpc'] == True:
    device = torch.device('cuda')
    planner = MPC(device, dvae, feature_dynamics_network, exp_args['action_horizon'], exp_args['n_actions'], exp_args['a_dim'], sampler='random')
elif exp_args['cem'] == True:
    device = torch.device('cuda')
    planner = CEM(device, dvae, feature_dynamics_network, exp_args['action_horizon'], exp_args['n_actions'], exp_args['a_dim'], sampler='random')

pred_next_state = planner.dynamics_model(state, a)

print("OG CD: ", chamfer_distance(torch.from_numpy(np.expand_dims(state, axis=0)).cuda(), torch.from_numpy(np.expand_dims(gt_ns, axis=0)).cuda())[0].cpu().detach().numpy())
print("Pred CD: ", chamfer_distance(torch.from_numpy(np.expand_dims(pred_next_state, axis=0)).cuda(), torch.from_numpy(np.expand_dims(gt_ns, axis=0)).cuda())[0].cpu().detach().numpy())

chamferDist = ChamferDistance()
print("OG CD: ", chamferDist(torch.from_numpy(np.expand_dims(state, axis=0)).cuda(), torch.from_numpy(np.expand_dims(gt_ns, axis=0)).cuda()).detach().cpu().item())
print("Pred CD: ", chamferDist(torch.from_numpy(np.expand_dims(pred_next_state, axis=0)).cuda(), torch.from_numpy(np.expand_dims(gt_ns, axis=0)).cuda()).detach().cpu().item())

# create og full point cloud [BLUE]
ns = gt_ns
og_pcl = o3d.geometry.PointCloud()
og_pcl.points = o3d.utility.Vector3dVector(ns)
og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

# visualize state cloud
s = state
s_pcl = o3d.geometry.PointCloud()
s_pcl.points = o3d.utility.Vector3dVector(s)
s_colors = np.tile(np.array([0, 1, 0]), (len(s),1))
s_pcl.colors = o3d.utility.Vector3dVector(s_colors)

# visualize reconstructed full cloud cloud [RED]
recon_pcl = pred_next_state
recon_pcl = np.reshape(recon_pcl, (2048, 3))
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(recon_pcl)
pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
o3d.visualization.draw_geometries([pcl, og_pcl, s_pcl])