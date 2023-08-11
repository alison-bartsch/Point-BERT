from models.dvae import *
from tools import builder
from utils.config import cfg_from_yaml_file
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join, exists

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
# from pytorch3d.loss import chamfer_distance # TODO: perhaps use this repository's version of CD????
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from pytorch3d.loss import chamfer_distance
from torch_geometric.data import Data, Batch
# from knn_cuda import KNN
from torch_cluster import knn_graph

# import utils as utils # TODO: add in utils file in dynamics folder and change input
# import emd.emd_module as emd
# import dynamics_model as dynamics # TODO: change import structure for dynamics model
# from action_dataset import ActionDataset # TODO: change import structure for dynamics dataset
from dynamics import dynamics_utils as utils
from dynamics import dynamics_model as dynamics
from dynamics.dynamics_dataset import DemoActionDataset, DemoWordDataset

def get_dataloaders(pcl_type, dvae=None):
    """
    Insert comment
    """
    # full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/May10_5D', pcl_type)
    full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/All_Shapes', pcl_type)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size])
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def return_nearest_neighbor(pcl_batch):
    """
    Given a batch of point clouds efficiently return the distance between each point
    and its nearest neighbor in the same point cloud.

    input size: (batch size, n points, 3)
    output size: (batch size, n points)
    """
    dist = torch.cdist(pcl_batch, pcl_batch)
    dist.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    new_dist = dist.clone().detach() 
    nn_dists = torch.min(new_dist, dim=2).values
    return nn_dists

def spacing_loss(pcl_batch, gate_dist, device):
    """
    Loss to discourage points in point cloud getting spaced too close to each other
    with the CD loss.
    """
    gate_dists = gate_dist * torch.ones(pcl_batch.size()[0], pcl_batch.size()[1]).to(device)
    nn_dists = return_nearest_neighbor(pcl_batch)
    relu = torch.nn.ReLU()
    point_wise_loss = torch.square(relu(gate_dists - nn_dists))
    batch_loss = torch.sum(point_wise_loss, dim = 1)
    # average across the batch
    loss = torch.mean(batch_loss)
    return loss


def train_feature_dynamics(dvae, feature_dynamics_network, optimizer, scheduler, train_loader, epoch, device):
    dvae.eval()
    feature_dynamics_network.train()

    stats = utils.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    parameters = list(feature_dynamics_network.parameters())
    for states, next_states, actions in train_loader:

        states = states.cuda()
        next_states = states.cuda()
        actions = actions.cuda()

        states_sampled, states_neighborhood, states_center, states_logits = dvae.encode(states) #.to(device)

        _, _, ns_center, _ = dvae.encode(next_states) #.to(device)
        pred_features = feature_dynamics_network(states_sampled, ns_center, actions)

        ret = dvae.decode_features(pred_features, states_neighborhood, ns_center, states_logits, states)
        _, neighborhood_ns, _, _ = dvae.encode(next_states)
        combo_ret = (ret[0], ret[1], ret[2], ret[3], neighborhood_ns, ret[5])
        loss = dvae.recon_loss(combo_ret, next_states)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Feat Train Loss {avg_loss:.4f}')
        pbar.update(states.shape[0])
    pbar.close()
    scheduler.step()
    return stats

def test_feature_dynamics(dvae, feature_dynamics_network, optimizer, test_loader, epoch, device):
    dvae.eval()
    feature_dynamics_network.eval()

    test_loss = 0
    for states, next_states, actions in test_loader:
        with torch.no_grad():
            states = states.cuda()
            next_states = states.cuda()
            actions = actions.cuda()

            states_sampled, states_neighborhood, states_center, states_logits = dvae.encode(states) #.to(device)

            _, _, ns_center, _ = dvae.encode(next_states) #.to(device)
            pred_features = feature_dynamics_network(states_sampled, ns_center, actions)

            ret = dvae.decode_features(pred_features, states_neighborhood, ns_center, states_logits, states)
            _, neighborhood_ns, _, _ = dvae.encode(next_states)
            combo_ret = (ret[0], ret[1], ret[2], ret[3], neighborhood_ns, ret[5])
            loss = dvae.recon_loss(combo_ret, next_states)

        test_loss += loss * states.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Feat Test Loss: {test_loss:.4f}')
    return test_loss.item()

def main(exp_name, delta=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('feature_experiments', exp_name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    writer = SummaryWriter(join(folder_name, 'data'))

    style_dict = {
        'delta' : delta
    }

    save_args = vars(args)
    save_args['script'] = 'train_dynamics'
    with open(join(folder_name, 'params.json'), 'w') as f:
        json.dump(style_dict, f)
        json.dump(save_args, f) 

    device = torch.device('cuda')

    # load the dynamics models
    token_dims = 256
    decoder_dims = 256
    n_tokens = 64
    feature_dynamics_network = dynamics.DGCNNDynamics(args.a_dim, token_dims, decoder_dims, n_tokens).to(device)

    parameters = list(feature_dynamics_network.parameters())
   
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                    # milestones=[200, 250, 300, 350, 400, 450],
                    milestones=[15, 50, 100, 400],
                    # milestones=[200, 300],
                    # milestones=[400, 500, 600, 700, 800, 900],
                    gamma=0.5)

    # load the pre-trained dvae
    config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
    config=config.config
    model_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth' # set correct model path to load from
    dvae = builder.model_builder(config)
    builder.load_model(dvae, model_path, logger = 'dvae_testclay')
    dvae.to(device)
    # dvae.to(0)

    train_loader, test_loader = get_dataloaders(args.pcl_type, dvae)

    print('--------------------------- Training Feature Dynamics ---------------------------')
    best_test_loss = float('inf')
    itr = 0
    for epoch in range(args.epochs):
        stats = train_feature_dynamics(dvae, feature_dynamics_network, optimizer, scheduler, train_loader, epoch, device)
        test_loss = test_feature_dynamics(dvae, feature_dynamics_network, optimizer, test_loader, epoch, device)

        # Log metrics
        old_itr = itr
        for k, values in stats.items():
            itr = old_itr
            for v in values:
                writer.add_scalar(k, v, itr)
                itr += 1
        writer.add_scalar('test_loss', test_loss, epoch)

        if epoch % args.log_interval == 0:
            if test_loss <= best_test_loss:
                best_test_loss = test_loss

                with open(folder_name + '/best_feat_test_loss.txt', 'w') as f:
                    string = str(epoch) + ':   ' + str(test_loss)
                    f.write(string)
                
                checkpoint = {
                    'feature_dynamics_network': feature_dynamics_network,
                    'optimizer': optimizer,
                }

                torch.save(checkpoint, join(folder_name, 'checkpoint'))
                print('Saved models with loss', best_test_loss)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, help='default 0')
    parser.add_argument('--epochs', type=int, default=500, help='default: 100') # 500
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--batch_size', type=int, default=16, help='default 32') # 32

    # Action and Cloud Parameters
    parser.add_argument('--a_dim', type=int, default=5, help='dimension of the action')
    parser.add_argument('--n_pts', type=int, default=2048, help='number of points in point cloud') 
    parser.add_argument('--pcl_type', type=str, default='shell_scaled', help='options: dense_centered, dense_scaled, shell_centered, shell_scaled')
 
    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='exp1', help='folder name results are stored into')
    args = parser.parse_args()


    # training styles: 'independent', 'sequential', 'gan'
    # main('independent', 'exp1')
    # main('sequential', 'exp2')
    main('exp2', delta=True)