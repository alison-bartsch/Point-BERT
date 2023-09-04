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

path = '/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos'
dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
device = torch.device('cuda')

# for the models that were trained separately
center_dynamics_path = 'centroid_experiments/exp33_geometric' # 'dvae_dynamics_experiments/exp16_center_pointnet'
feature_dynamics_path = 'feature_experiments/exp15_new_dataset' # 'dvae_dynamics_experiments/exp39_dgcnn_pointnet'
checkpoint = torch.load(feature_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
feature_dynamics_network = checkpoint['feature_dynamics_network'].to(device) # 'dynamics_network'
ctr_checkpoint = torch.load(center_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
center_dynamics_network = ctr_checkpoint['center_dynamics_network'].to(device)

# load the dvae model
config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
config=config.config
dvae = builder.model_builder(config)
builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
dvae.to(device)
dvae.eval()

# initialize the dataset for feature dynamics
dataset = DemoActionDataset(path, 'shell_scaled')
geometric_dataset = GeometricDataset(path, 'shell_scaled')

# track chamfer distance
cds_full_cloud = []
cds_centroid = []

# loop for iterating through all the samples, and reporting:
    # (1) mean + std dev of CD on whole pcl
    # (2) mean + std dev of CD on centroid
for index in tqdm(range(7620)):
    state, next_state, action = dataset.__getitem__(index)
    state = torch.unsqueeze(state, 0)
    next_state = torch.unsqueeze(next_state, 0)
    action = torch.unsqueeze(action, 0)

    state = state.cuda()
    next_state = next_state.cuda()
    action = action.cuda()

    z_states, neighborhood, center, logits = dvae.encode(state) #.to(device)
    z_ns, ns_neighborhood, ns_gt_center, ns_logits = dvae.encode(next_state)
    ns_vae_decoded = dvae.decode(z_ns, ns_neighborhood, ns_gt_center, ns_logits, next_state)

    unnormalized_state, unnormalized_ns, unnormalized_action, normalization_centroid, _ = geometric_dataset.__getitem__(index)
    ns_center_unnormalized = predict_centroid_dynamics(unnormalized_state, unnormalized_action)

    ns_center = ns_center_unnormalized - normalization_centroid
    m = np.max(np.sqrt(np.sum(ns_center**2, axis=1)))
    ns_center = ns_center / m
    ns_center = torch.from_numpy(ns_center).float().unsqueeze(0).cuda() 

    # feature dynamics prediction
    pred_features = feature_dynamics_network(z_states, ns_center, action)

    # decode the prediction
    ret_recon_next = dvae.decode_features(pred_features, neighborhood, ns_center, logits, state)
    recon_pcl = ret_recon_next[1]

    # get chamfer distance between recon_pcl and ns 
    chamfer_full_cloud = chamfer_distance(next_state, ret_recon_next[1])[0].cpu().detach().numpy()
    chamfer_centroid = chamfer_distance(ns_gt_center, ns_center)[0].cpu().detach().numpy()

    cds_full_cloud.append(chamfer_full_cloud)
    cds_centroid.append(chamfer_centroid)

mean_cd = np.mean(cds_full_cloud)
std_cd = np.std(cds_full_cloud)
print("CD Full cloud mean: ", mean_cd)
print("CD Full cloud std: ", std_cd)

mean_cd_centroid = np.mean(cds_centroid)
std_cd_centroid = np.std(cds_centroid)
print("CD Centroid mean: ", mean_cd_centroid)
print("CD Centroid std: ", std_cd_centroid)

assert False

test_samples = [2, 2133, 6011, 0, 60, 120, 180, 240, 300, 360, 420, 480, 3060] # , 3122, 6011, 7048]
for index in test_samples:
    state, next_state, action = dataset.__getitem__(index)
    state = torch.unsqueeze(state, 0)
    next_state = torch.unsqueeze(next_state, 0)
    action = torch.unsqueeze(action, 0)

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

    # TODO: add in the geometric centroid dynamics propagator
        # should take in the centroid state (unnormalized) and the action
        # should return ns_center
    unnormalized_state, unnormalized_ns, unnormalized_action, normalization_centroid, _ = geometric_dataset.__getitem__(index)
    ns_center_unnormalized = predict_centroid_dynamics(unnormalized_state, unnormalized_action)

    # TODO: make this optional
    ns_center_torch = torch.from_numpy(ns_center_unnormalized).unsqueeze(0).cuda()
    ns_center_delta = center_dynamics_network(ns_center_torch, action)
    ns_center_delta = ns_center_delta.detach().cpu().numpy()
    # print("ns_center_delta shape: ", ns_center_delta.shape)

    # ns_center_unnormalized = ns_center_unnormalized + ns_center_delta # OPTIONAL
    # print("ns center unnormalized: ", ns_center_unnormalized.shape)

    ns_center = ns_center_unnormalized - normalization_centroid
    m = np.max(np.sqrt(np.sum(ns_center**2, axis=1)))
    ns_center = ns_center / m
    # ns_center = ns_center + ns_center_delta
    ns_center = torch.from_numpy(ns_center).float().unsqueeze(0).cuda() 
    # ns_center = torch.from_numpy(ns_center).float().cuda() 
    print("ns_center: ", ns_center.size())

    # feature dynamics prediction
    pred_features = feature_dynamics_network(z_states, ns_center, action)

    # decode the prediction
    ret_recon_next = dvae.decode_features(pred_features, neighborhood, ns_center, logits, state)
    recon_pcl = ret_recon_next[1]

    # visualize state cloud
    s = state.squeeze().detach().cpu().numpy()
    s_pcl = o3d.geometry.PointCloud()
    s_pcl.points = o3d.utility.Vector3dVector(s)
    s_colors = np.tile(np.array([0, 0, 1]), (len(s),1))
    s_pcl.colors = o3d.utility.Vector3dVector(s_colors)

    # visualize reconstructed full cloud cloud [RED]
    recon_pcl = recon_pcl.squeeze().detach().cpu().numpy()
    recon_pcl = np.reshape(recon_pcl, (2048, 3))
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(recon_pcl)
    pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
    pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
    o3d.visualization.draw_geometries([pcl, og_pcl]) # , vae_pcl])
    o3d.visualization.draw_geometries([s_pcl])
    o3d.visualization.draw_geometries([pcl])
    o3d.visualization.draw_geometries([og_pcl])

    # # create og centroid point cloud [BLUE]
    # # ns_center = ns_gt_center.squeeze().detach().cpu().numpy()
    # ns_center_pcl = ns_gt_center.squeeze().cpu().numpy()
    # og_pcl_cent = o3d.geometry.PointCloud()
    # og_pcl_cent.points = o3d.utility.Vector3dVector(ns_center_pcl)
    # og_colors_cent = np.tile(np.array([0, 0, 1]), (len(ns_center_pcl),1))
    # og_pcl_cent.colors = o3d.utility.Vector3dVector(og_colors_cent)

    # # create state centroid point cloud [GREEN]
    # s_center_pcl = center.squeeze().cpu().numpy()
    # s_pcl = o3d.geometry.PointCloud()
    # s_pcl.points = o3d.utility.Vector3dVector(s_center_pcl)
    # s_colors = np.tile(np.array([0, 1, 0]), (len(s_center_pcl),1))
    # s_pcl.colors = o3d.utility.Vector3dVector(s_colors)
    
    # # visualize reconstructed centoird point cloud [RED]
    # pred_ns_center = ns_center.squeeze().detach().cpu().numpy()
    # ns_pcl_cent = o3d.geometry.PointCloud()
    # ns_pcl_cent.points = o3d.utility.Vector3dVector(pred_ns_center)
    # ns_pcl_colors = np.tile(np.array([1, 0, 0]), (len(pred_ns_center),1))
    # ns_pcl_cent.colors = o3d.utility.Vector3dVector(ns_pcl_colors)
    # o3d.visualization.draw_geometries([s_pcl, og_pcl_cent, ns_pcl_cent])

    # get chamfer distance between recon_pcl and ns 
    chamfer_full_cloud = chamfer_distance(next_state, ret_recon_next[1])[0].cpu().detach().numpy()
    print("CD: ", chamfer_full_cloud)
    
    cds_full_cloud.append(chamfer_full_cloud)

mean_cd = np.mean(cds_full_cloud)
print("Mean CD: ", mean_cd)