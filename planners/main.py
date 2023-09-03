import numpy as np


# start by naming the experiment and the experiment parameters
# create a folder with the experiment name and save a json file with the experiment parameters
# create two threads: one for the external camera to collect a video and one for the planning / communitation with robot computer

# MAIN THREAD:

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