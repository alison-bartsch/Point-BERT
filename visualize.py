import torch
import numpy as np
import open3d as o3d
from models.dvae import *
from tools import builder
from utils.config import cfg_from_yaml_file
from dynamics.dynamics_dataset import DemoActionDataset, DemoWordDataset

"""
Reconstructing next state with state's centers
"""
# path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'
# dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'

# # load the dvae model
# config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
# config=config.config
# dvae = builder.model_builder(config)
# builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
# device = torch.device('cuda')
# dvae.to(device)
# dvae.eval()

# # initialize the dataset
# dataset = DemoActionDataset(path, 'shell_scaled')

# # iterate through a few state/next state pairs
# test_samples = [0, 60, 120, 180, 240, 300, 360] # [0, 6, 100, 3000, 5067, 2048, 2678, 3333, 6983, 222, 468, 172]
# for index in test_samples:
#     state, next_state, action = dataset.__getitem__(index)
#     state = torch.unsqueeze(state, 0)
#     next_state = torch.unsqueeze(next_state, 0)

#     # create og point cloud
#     ns = next_state.squeeze().detach().cpu().numpy()
#     og_pcl = o3d.geometry.PointCloud()
#     og_pcl.points = o3d.utility.Vector3dVector(ns)
#     og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
#     og_pcl.colors = o3d.utility.Vector3dVector(og_colors)
#     # o3d.visualization.draw_geometries([og_pcl])

#     state = state.cuda()
#     next_state = next_state.cuda()

#     # reconstruct next_state point cloud with state's centers
#     z_states, state_neighborhood, state_center, state_logits = dvae.encode(state) #.to(device)
#     # z_next_states, ns_neighborhood, ns_center, ns_logits = dvae.encode(next_state)
#     z_next_states, ns_neighborhood, ns_center, ns_logits = dvae.next_state_encode(next_state, state_center, state_neighborhood)
#     ret_recon_next = dvae.decode(z_next_states, ns_neighborhood, ns_center, ns_logits, state) #.to(device)

#     # ret_recon_next = dvae.decode(z_states, state_neighborhood, ns_center, ns_logits, state)
#     recon_pcl = ret_recon_next[1]

#     # visualize reconstructed cloud
#     recon_pcl = recon_pcl.squeeze().detach().cpu().numpy()
#     recon_pcl = np.reshape(recon_pcl, (2048, 3))
#     pcl = o3d.geometry.PointCloud()
#     pcl.points = o3d.utility.Vector3dVector(recon_pcl)
#     pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
#     pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
#     # o3d.visualization.draw_geometries([pcl])
#     o3d.visualization.draw_geometries([pcl, og_pcl])

# assert False

"""
Visualize center cluster dynamics model
"""
# define the action space and dynamics loss type
path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'
dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
word_dynamics_path = 'dvae_dynamics_experiments/exp8_twonetworks_ce'
center_dynamics_path = 'dvae_dynamics_experiments/exp10_centroid'

# load the dvae model
config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
config=config.config
dvae = builder.model_builder(config)
builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
device = torch.device('cuda')
dvae.to(device)
dvae.eval()

# load the checkpoint
checkpoint = torch.load(center_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
word_checkpoint = torch.load(word_dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
dynamics_network = checkpoint['dynamics_network'].to(device)
word_dynamics = word_checkpoint['dynamics_network'].to(device)

# initialize the dataset
dataset = DemoWordDataset(path, 'shell_scaled', dvae)

# do we also predict the new vocab?
predict_words = True

# test_samples = [0, 6, 100, 3000, 5067, 2048, 2678, 3333, 6983, 222, 468, 172]

# samples with the largest s vs ns difference
test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120]
for index in test_samples:
    state, next_state, action, _ = dataset.__getitem__(index)

    state = torch.unsqueeze(state, 0)
    next_state = torch.unsqueeze(next_state, 0)
    action = torch.unsqueeze(action, 0)

    # create gt next state point cloud [BLUE]
    ns = next_state.squeeze().detach().cpu().numpy()
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(ns)
    og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)
    # o3d.visualization.draw_geometries([og_pcl])

    # create gt state point cloud [GREEN]
    s = state.squeeze().detach().cpu().numpy()
    s_pcl = o3d.geometry.PointCloud()
    s_pcl.points = o3d.utility.Vector3dVector(s)
    s_colors = np.tile(np.array([0, 1, 0]), (len(s),1))
    s_pcl.colors = o3d.utility.Vector3dVector(s_colors)

    state = state.cuda()
    next_state = next_state.cuda()
    action = action.cuda()

    z_state, neighborhood, center, logits = dvae.encode(state) 
    ns_center_pred = dynamics_network(center, action).to(device)

    if predict_words:
    # --------- calculate new vocab ---------
        # TODO: INCLUDE SOME CLUSTERING/REORDERING OF NS_CENTER_PRED TO MATCH THE VOCAB ORDER OF THOSE NEARBY
        group_center = ns_center_pred.squeeze()
        vocab = z_state.squeeze()
        action = torch.tile(action, (vocab.size()[0], 1))

        ns_logits = word_dynamics(vocab, group_center, action)
        ns_logits = torch.unsqueeze(ns_logits, 0)
        print("\nns_logits.size():", ns_logits.size())
        latent_sampled = dvae.latent_logits_sample(ns_logits)

        ret_recon_next = dvae.decode(latent_sampled, neighborhood, ns_center_pred, ns_logits, state) #.to(device)
        recon_pcl = ret_recon_next[1]
    
    else:
        ret_recon_next = dvae.decode(z_state, neighborhood, ns_center_pred, logits, state) #.to(device)
        recon_pcl = ret_recon_next[1]

    # visualize reconstructed cloud [RED]
    recon_pcl = recon_pcl.squeeze().detach().cpu().numpy()
    recon_pcl = np.reshape(recon_pcl, (2048, 3))
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(recon_pcl)
    pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
    pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
    # o3d.visualization.draw_geometries([pcl])
    o3d.visualization.draw_geometries([og_pcl, pcl])
    # o3d.visualization.draw_geometries([pcl, s_pcl, og_pcl])

assert False


"""
Visualize word-level dynamics model
"""
# # define the action space and dynamics loss type
# path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'
# dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
# dynamics_path = 'dvae_dynamics_experiments/exp8_twonetworks_ce'

# # load the dvae model
# config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
# config=config.config
# dvae = builder.model_builder(config)
# builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
# device = torch.device('cuda')
# dvae.to(device)
# dvae.eval()

# # load the checkpoint
# checkpoint = torch.load(dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
# dynamics_network = checkpoint['dynamics_network'].to(device)

# # initialize the dataset
# dataset = DemoWordDataset(path, 'shell_scaled', dvae)

# # test_samples = [0, 6, 100, 3000, 5067, 2048, 2678, 3333, 6983, 222, 468, 172]

# # samples with the largest s vs ns difference
# test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120]
# for index in test_samples:
#     state, next_state, action, _ = dataset.__getitem__(index)

#     state = torch.unsqueeze(state, 0)
#     next_state = torch.unsqueeze(next_state, 0)
#     action = torch.unsqueeze(action, 0)

#     # create gt next state point cloud [BLUE]
#     ns = next_state.squeeze().detach().cpu().numpy()
#     og_pcl = o3d.geometry.PointCloud()
#     og_pcl.points = o3d.utility.Vector3dVector(ns)
#     og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
#     og_pcl.colors = o3d.utility.Vector3dVector(og_colors)
#     # o3d.visualization.draw_geometries([og_pcl])

#     # create gt state point cloud [GREEN]
#     s = state.squeeze().detach().cpu().numpy()
#     s_pcl = o3d.geometry.PointCloud()
#     s_pcl.points = o3d.utility.Vector3dVector(s)
#     s_colors = np.tile(np.array([0, 1, 0]), (len(s),1))
#     s_pcl.colors = o3d.utility.Vector3dVector(s_colors)

#     state = state.cuda()
#     next_state = next_state.cuda()
#     action = action.cuda()

#     z_state, neighborhood, center, _ = dvae.encode(state) 
#     gt_z_next_state, _, _, _ = dvae.next_state_encode(next_state, center, neighborhood)

#     # ns_word = gt_z_next_state.squeeze()
#     # target = dvae.codebook_onehot(ns_word).cuda()
#     group_center = center.squeeze()
#     vocab = z_state.squeeze()
#     action = torch.tile(action, (vocab.size()[0], 1))

#     ns_logits = dynamics_network(vocab, group_center, action)
#     ns_logits = torch.unsqueeze(ns_logits, 0)
#     print("\nns_logits.size():", ns_logits.size())
#     latent_sampled = dvae.latent_logits_sample(ns_logits)

#     ret_recon_next = dvae.decode(latent_sampled, neighborhood, center, ns_logits, state) #.to(device)
#     recon_pcl = ret_recon_next[1]

#     # visualize reconstructed cloud [RED]
#     recon_pcl = recon_pcl.squeeze().detach().cpu().numpy()
#     recon_pcl = np.reshape(recon_pcl, (2048, 3))
#     pcl = o3d.geometry.PointCloud()
#     pcl.points = o3d.utility.Vector3dVector(recon_pcl)
#     pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
#     pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
#     # o3d.visualization.draw_geometries([pcl])
#     o3d.visualization.draw_geometries([og_pcl, pcl])
#     # o3d.visualization.draw_geometries([pcl, s_pcl, og_pcl])

# assert False



"""
Visualize the reconstructed predition of the dynamics model
"""
# # define the action space and dynamics loss type
# path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'
# dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'
# dynamics_path = 'dvae_dynamics_experiments/exp6_twonetworks_mse' # exp3_twonetworks_mse' # exp1_twonetworks'
# # actions = np.load(path + '/action_normalized.npy')

# # load the dvae model
# config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
# config=config.config
# dvae = builder.model_builder(config)
# builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
# device = torch.device('cuda')
# dvae.to(device)
# dvae.eval()

# # load the checkpoint
# checkpoint = torch.load(dynamics_path + '/checkpoint', map_location=torch.device('cpu'))
# dynamics_network = checkpoint['dynamics_network'].to(device)

# # initialize the dataset
# dataset = DemoActionDataset(path, 'shell_scaled')

# # iterate through a few state/next state pairs
# # test_samples = [0, 6, 100, 3000, 5067, 2048, 2678, 3333, 6983, 222, 468, 172]
# test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120]
# for index in test_samples:
#     state, next_state, action = dataset.__getitem__(index)
#     state = torch.unsqueeze(state, 0)
#     next_state = torch.unsqueeze(next_state, 0)
#     action = torch.unsqueeze(action, 0)

#     # create og point cloud
#     ns = next_state.squeeze().detach().cpu().numpy()
#     og_pcl = o3d.geometry.PointCloud()
#     og_pcl.points = o3d.utility.Vector3dVector(ns)
#     og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
#     og_pcl.colors = o3d.utility.Vector3dVector(og_colors)
#     # o3d.visualization.draw_geometries([og_pcl])

#     state = state.cuda()
#     next_state = next_state.cuda()
#     action = action.cuda()

#     z_states, neighborhood, center, logits = dvae.encode(state) #.to(device)
#     print("\nz states shape: ", z_states.size())
#     z_pred_next_states = dynamics_network(z_states, action).to(device)
#     ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, state) #.to(device)
#     recon_pcl = ret_recon_next[1]

#     # visualize reconstructed cloud
#     recon_pcl = recon_pcl.squeeze().detach().cpu().numpy()
#     recon_pcl = np.reshape(recon_pcl, (2048, 3))
#     pcl = o3d.geometry.PointCloud()
#     pcl.points = o3d.utility.Vector3dVector(recon_pcl)
#     pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
#     pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
#     # o3d.visualization.draw_geometries([pcl])
#     o3d.visualization.draw_geometries([pcl, og_pcl])


"""
Visualize the centers of state with the same vocab word
"""
# # define the action space and dynamics loss type
# path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'
# dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'

# # load the dvae model
# config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
# config=config.config
# dvae = builder.model_builder(config)
# builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
# device = torch.device('cuda')
# dvae.to(device)
# dvae.eval()

# # initialize the dataset
# dataset = DemoActionDataset(path, 'shell_scaled')

# # iterate through a few state/next state pairs
# # test_samples = [0, 6, 100, 3000, 5067, 2048, 2678, 3333, 6983, 222, 468, 172]

# state, next_state, action = dataset.__getitem__(0)
# state = torch.unsqueeze(state, 0)
# next_state = torch.unsqueeze(next_state, 0)
# action = torch.unsqueeze(action, 0)

# state = state.cuda()
# next_state = next_state.cuda()
# action = action.cuda()

# z_states, s_neighborhood, s_center, s_logits = dvae.encode(state) 

# # test_words = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120]
# for index in range(8192):
#     word = dvae.get_word(index)
#     same_vocab = torch.tile(word, (z_states.size()[1],1))
#     same_vocab = torch.unsqueeze(same_vocab, 0)
#     ret_recon_next = dvae.decode(same_vocab, s_neighborhood, s_center, s_logits, state)
#     recon_pcl = ret_recon_next[1]

#     # create og point cloud
#     s = state.squeeze().detach().cpu().numpy()
#     og_pcl = o3d.geometry.PointCloud()
#     og_pcl.points = o3d.utility.Vector3dVector(s)
#     og_colors = np.tile(np.array([0, 0, 1]), (len(s),1))
#     og_pcl.colors = o3d.utility.Vector3dVector(og_colors)
#     # o3d.visualization.draw_geometries([og_pcl])

#      # visualize reconstructed cloud
#     recon_pcl = recon_pcl.squeeze().detach().cpu().numpy()
#     recon_pcl = np.reshape(recon_pcl, (2048, 3))
#     pcl = o3d.geometry.PointCloud()
#     pcl.points = o3d.utility.Vector3dVector(recon_pcl)
#     pcl_colors = np.tile(np.array([1, 0, 0]), (len(recon_pcl),1))
#     pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
#     # o3d.visualization.draw_geometries([pcl])
#     o3d.visualization.draw_geometries([pcl, og_pcl])

# assert False


"""
Visualize the centers of state and next states
"""
# define the action space and dynamics loss type
path = '/home/alison/Clay_Data/Fully_Processed/All_Shapes'
dvae_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth'

# load the dvae model
config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
config=config.config
dvae = builder.model_builder(config)
builder.load_model(dvae, dvae_path, logger = 'dvae_testclay')
device = torch.device('cuda')
dvae.to(device)
dvae.eval()

# initialize the dataset
dataset = DemoActionDataset(path, 'shell_scaled')

# iterate through a few state/next state pairs
# test_samples = [0, 6, 100, 3000, 5067, 2048, 2678, 3333, 6983, 222, 468, 172]
test_samples = [0, 60, 120, 180, 240, 300, 360, 420, 480, 3000, 3060, 3120]
for index in test_samples:
    state, next_state, action = dataset.__getitem__(index)
    state = torch.unsqueeze(state, 0)
    next_state = torch.unsqueeze(next_state, 0)
    action = torch.unsqueeze(action, 0)

    state = state.cuda()
    next_state = next_state.cuda()
    action = action.cuda()

    z_states, s_neighborhood, s_center, s_logits = dvae.encode(state) #.to(device)
    z_next_states, ns_neighborhood, ns_center, ns_logits = dvae.encode(next_state)

    # print("\nz states shape: ", z_states.size())
    # z_pred_next_states = dynamics_network(z_states, action).to(device)
    # ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, state) #.to(device)
    # recon_pcl = ret_recon_next[1]



    # create the ns centers [BLUE]
    ns = ns_center.squeeze().detach().cpu().numpy()
    print("\ns shape: ", ns.shape)
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(ns)
    og_colors = np.tile(np.array([0, 0, 1]), (len(ns),1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)
    # o3d.visualization.draw_geometries([og_pcl])

    # create teh s centers [RED]
    s = s_center.squeeze().detach().cpu().numpy()
    # recon_pcl = np.reshape(recon_pcl, (2048, 3))
    print("s shape: ", s.shape)
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(s)
    pcl_colors = np.tile(np.array([1, 0, 0]), (len(s),1))
    pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
    # o3d.visualization.draw_geometries([pcl])
    o3d.visualization.draw_geometries([pcl, og_pcl])


    # # create the s centers [RED]
    # s = s_center.squeeze().detach().cpu().numpy()
    # ns = ns_center.squeeze().detach().cpu().numpy()

    # for i in range(s.shape[0]):
    #     s_i = np.reshape(s[i,:], (1,3))
    #     ns_i = np.reshape(ns[i,:], (1,3))
        
    #     np_pcl = np.vstack((s_i, ns_i))
    #     print("\nnp_pcl shape: ", np_pcl.shape)

    #     # og_pcl = o3d.geometry.PointCloud()
    #     # og_pcl.points = o3d.utility.Vector3dVector(ns_i)
    #     # og_colors = np.tile(np.array([0, 0, 1]), (len(ns_i),1))
    #     # og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

    #     og_pcl = o3d.geometry.PointCloud()
    #     og_pcl.points = o3d.utility.Vector3dVector(s)
    #     og_colors = np.tile(np.array([0, 0, 1]), (len(s),1))
    #     og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

    #     pcl = o3d.geometry.PointCloud()
    #     pcl.points = o3d.utility.Vector3dVector(np_pcl)
    #     pcl_colors = np.tile(np.array([1, 0, 0]), (len(np_pcl),1))
    #     pcl.colors = o3d.utility.Vector3dVector(pcl_colors)
    #     o3d.visualization.draw_geometries([og_pcl, pcl])
