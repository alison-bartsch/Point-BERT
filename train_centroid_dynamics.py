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
from dynamics.dynamics_dataset import DemoActionDataset, GeometricDataset
from geometric_utils import *

def get_dataloaders(pcl_type, geometric=True, dvae=None):
    """
    Insert comment
    """
    # full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/May10_5D', pcl_type)
    # full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/All_Shapes', pcl_type)
    # full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/Aug24_Human_Demos_Fully_Processed', pcl_type)
    
    if geometric:
        full_dataset = GeometricDataset('/home/alison/Clay_Data/Fully_Processed/Aug29_Human_Demos', pcl_type)
    else:
        full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/Aug29_Human_Demos', pcl_type)
    
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

def train_center_dynamics(dvae, center_dynamics_network, optimizer, scheduler, train_loader, epoch, device, delta):
    dvae.eval()
    center_dynamics_network.train()

    stats = utils.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    parameters = list(center_dynamics_network.parameters())
    for states, next_states, actions in train_loader:

        states = states.cuda()
        next_states = next_states.cuda()
        actions = actions.cuda()

        _, _, center_state, _ = dvae.encode(states) #.to(device)
        _, _, center_next_states, _ = dvae.encode(next_states)

        if delta:
            # # get nearest neighbors between center_state and center_next_states
            # # get the distance between each, and that is the delta (order matters in this case -- MSE loss???)
            # ns_delta = center_dynamics_network(center_state, actions).to(device)
            # # TODO: modify this to get the nearest neighbors so it's not arbitrary
            # # gt_delta = center_next_states - center_state
            # # loss_func = nn.MSELoss()
            # # loss = loss_func(ns_delta, gt_delta)

            # # loss alternative: 
            # ns_center_pred = center_state + ns_delta
            # loss = chamfer_distance(center_next_states, ns_center_pred)[0]

            # ns_delta = center_dynamics_network(states, actions).to(device)
            # ns_pred = states + ns_delta
            # loss = chamfer_distance(next_states, ns_pred)[0]
            ns_center_delta = center_dynamics_network(center_state, actions).to(device)

            # # TODO: try predicting deltas directly
            # gt_residuals = center_next_states - center_state
            # loss = chamfer_distance(ns_center_delta, gt_residuals)[0]

            # THIS IS THE OG DELTAS, THEY WERE COLLAPSING TO 0
            ns_center_pred = center_state + ns_center_delta
            loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        else:
            ns_center_pred = center_dynamics_network(center_state, actions).to(device)
            loss = chamfer_distance(center_next_states, ns_center_pred)[0]

            # loss_func = nn.MSELoss()
            # loss = loss_func(center_next_states, ns_center_pred)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Ctr Train Loss {avg_loss:.4f}')
        pbar.update(states.shape[0])
    pbar.close()
    scheduler.step()
    return stats

def test_center_dynamics(dvae, center_dynamics_network, optimizer, test_loader, epoch, device, delta):
    dvae.eval()
    center_dynamics_network.eval()

    test_loss = 0
    for states, next_states, actions in test_loader:

        states = states.cuda()
        next_states = next_states.cuda()
        actions = actions.cuda()

        _, _, center_state, _ = dvae.encode(states) #.to(device)
        _, _, center_next_states, _ = dvae.encode(next_states)

        if delta:
            # # get nearest neighbors between center_state and center_next_states
            # # get the distance between each, and that is the delta (order matters in this case -- MSE loss???)
            # ns_delta = center_dynamics_network(center_state, actions).to(device)
            # # TODO: modify this to get the nearest neighbors so it's not arbitrary
            # # gt_delta = center_next_states - center_state
            # # loss_func = nn.MSELoss()
            # # loss = loss_func(ns_delta, gt_delta)

            # # loss alternative:
            # ns_center_pred = center_state + ns_delta
            # loss = chamfer_distance(center_next_states, ns_center_pred)[0]

            # ns_delta = center_dynamics_network(states, actions).to(device)
            # ns_pred = states + ns_delta
            # loss = chamfer_distance(next_states, ns_pred)[0]
            ns_center_delta = center_dynamics_network(center_state, actions).to(device)

            # # TODO: try predicting deltas directly
            # gt_residuals = center_next_states - center_state
            # loss = chamfer_distance(ns_center_delta, gt_residuals)[0]

            # THIS IS THE OG DELTAS, THEY WERE COLLAPSING TO 0
            ns_center_pred = center_state + ns_center_delta
            loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        else:
            ns_center_pred = center_dynamics_network(center_state, actions).to(device)
            loss = chamfer_distance(center_next_states, ns_center_pred)[0]

            # loss_func = nn.MSELoss()
            # loss = loss_func(center_next_states, ns_center_pred)

        # ns_center_pred = center_dynamics_network(center_state, actions).to(device)
        # loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        test_loss += loss * states.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Ctr Test Loss: {test_loss:.4f}')
    return test_loss.item()


def train_geometric_center_dynamics(dvae, center_dynamics_network, optimizer, scheduler, train_loader, epoch, device, delta):
    dvae.eval()
    center_dynamics_network.train()

    stats = utils.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    parameters = list(center_dynamics_network.parameters())
    for states, next_states, actions, s_centroid, ns_centroid in train_loader:

        states = states.cuda()
        next_states = next_states.cuda()
        actions = actions.cuda()

        _, _, center_state, _ = dvae.encode(states) #.to(device)
        _, _, center_next_states, _ = dvae.encode(next_states)

        if delta:
            # pass through geometric centroid propagation
            ns_ctr_list = []
            for i in range(center_state.shape[0]):
                s_ctr = center_state[i].detach().cpu()
                a = actions[i].cpu() # .detach().cpu().numpy()

                # predict next state centroids with geometry
                ns_center_unnormalized = predict_centroid_dynamics(s_ctr, a)

                # # normalize prediction for comparison
                # ns_center_unnormalized = ns_center_unnormalized - s_centroid[i].cpu()
                # m = np.max(np.sqrt(np.sum(ns_center_unnormalized**2, axis=1)))
                # ns_center_elem = ns_center_unnormalized / m
                # ns_center_elem = torch.from_numpy(ns_center_elem).float().cuda()
                # ns_ctr_list.append(ns_center_elem)

                ns_center_elem = torch.from_numpy(ns_center_unnormalized).float().cuda()
                ns_ctr_list.append(ns_center_elem)
            
            geometric_ns = torch.stack(ns_ctr_list)
            ns_center_delta = center_dynamics_network(geometric_ns, actions).to(device)
            ns_center_pred = geometric_ns + ns_center_delta

            # # normalize the center next states
            # center_next_states = center_next_states - ns_centroid
            # m = torch.max(torch.sqrt(torch.sum(center_next_states**2, dim=1)))
            # center_next_states = center_next_states / m

            loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        else:
            ns_center_pred = center_dynamics_network(center_state, actions).to(device)
            loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Ctr Train Loss {avg_loss:.4f}')
        pbar.update(states.shape[0])
    pbar.close()
    scheduler.step()
    return stats

def test_geometric_center_dynamics(dvae, center_dynamics_network, optimizer, test_loader, epoch, device, delta):
    dvae.eval()
    center_dynamics_network.eval()

    test_loss = 0
    for states, next_states, actions, s_centroids, ns_centroids in test_loader:

        states = states.cuda()
        next_states = next_states.cuda()
        actions = actions.cuda()

        _, _, center_state, _ = dvae.encode(states) #.to(device)
        _, _, center_next_states, _ = dvae.encode(next_states)

        if delta:
            # pass through geometric centroid propagation
            ns_ctr_list = []
            for i in range(center_state.shape[0]):
                s_ctr = center_state[i].detach().cpu()
                a = actions[i].cpu() # .detach().cpu().numpy()

                # predict next state centroids with geometry
                ns_center_unnormalized = predict_centroid_dynamics(s_ctr, a)

                # # normalize prediction for comparison
                # ns_center_unnormalized = ns_center_unnormalized - s_centroid[i].cpu()
                # m = np.max(np.sqrt(np.sum(ns_center_unnormalized**2, axis=1)))
                # ns_center_elem = ns_center_unnormalized / m
                # ns_center_elem = torch.from_numpy(ns_center_elem).float().cuda()
                # ns_ctr_list.append(ns_center_elem)

                ns_center_elem = torch.from_numpy(ns_center_unnormalized).float().cuda()
                ns_ctr_list.append(ns_center_elem)
            
            geometric_ns = torch.stack(ns_ctr_list)
            ns_center_delta = center_dynamics_network(geometric_ns, actions).to(device)
            ns_center_pred = geometric_ns + ns_center_delta

            # # normalize the center next states
            # center_next_states = center_next_states - ns_centroid
            # m = torch.max(torch.sqrt(torch.sum(center_next_states**2, dim=1)))
            # center_next_states = center_next_states / m

            loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        else:
            ns_center_pred = center_dynamics_network(center_state, actions).to(device)
            loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        test_loss += loss * states.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Ctr Test Loss: {test_loss:.4f}')
    return test_loss.item()


def main(exp_name, geometric=True, delta=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('centroid_experiments', exp_name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    writer = SummaryWriter(join(folder_name, 'data'))

    style_dict = {
        'delta' : delta,
        'geometric': geometric
    }

    save_args = vars(args)
    save_args['script'] = 'train_dynamics'
    with open(join(folder_name, 'params.json'), 'w') as f:
        json.dump(style_dict, f)
        json.dump(save_args, f) 

    device = torch.device('cuda')

    # load the dynamics models
    dim = 64*3
    # dim = 1048*3
    input_dim = dim + args.a_dim
    if args.model_type == 'encoder':
        print("Using encoder model")
        center_dynamics_network = dynamics.CentroidDynamics(dim).to(device)
    elif args.model_type == 'pointnet_action':
        print("Using action encoder + pointnet model")
        center_dynamics_network = dynamics.PointNetActionDynamics(dim).to(device)
    else:
        center_dynamics_network = dynamics.PointNetDynamics(dim).to(device) # (input_dim).to(device)

    parameters = list(center_dynamics_network.parameters())
   
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            # milestones=[50, 100, 150, 200, 400],
                            # milestones=[10, 25, 40, 50, 75, 100, 125, 150, 200, 250, 300],
                            milestones=[75, 150, 200],
                    # milestones=[200, 250, 300, 350, 400, 450],
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

    print('--------------------------- Training Center Dynamics ---------------------------')
    best_test_loss = float('inf')
    itr = 0
    for epoch in range(args.epochs):
        if geometric:
            stats = train_geometric_center_dynamics(dvae, center_dynamics_network, optimizer, scheduler, train_loader, epoch, device, delta)
            test_loss = test_geometric_center_dynamics(dvae, center_dynamics_network, optimizer, test_loader, epoch, device, delta)
        else:
            stats = train_center_dynamics(dvae, center_dynamics_network, optimizer, scheduler, train_loader, epoch, device, delta)
            test_loss = test_center_dynamics(dvae, center_dynamics_network, optimizer, test_loader, epoch, device, delta)

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

                with open(folder_name + '/best_ctr_test_loss.txt', 'w') as f:
                    string = str(epoch) + ':   ' + str(test_loss)
                    f.write(string)
                
                checkpoint = {
                    'center_dynamics_network': center_dynamics_network,
                    'optimizer': optimizer,
                }

                torch.save(checkpoint, join(folder_name, 'checkpoint'))
                print('Saved models with loss', best_test_loss)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-6, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, help='default 0')
    parser.add_argument('--epochs', type=int, default=200, help='default: 100') # 500
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--batch_size', type=int, default=16, help='default 32') # 32

    # Action and Cloud Parameters
    parser.add_argument('--a_dim', type=int, default=5, help='dimension of the action')
    parser.add_argument('--n_pts', type=int, default=2048, help='number of points in point cloud') 
    parser.add_argument('--pcl_type', type=str, default='shell_scaled', help='options: dense_centered, dense_scaled, shell_centered, shell_scaled')
    parser.add_argument('--model_type', type=str, default='pointnet_encoder', help='options are encoder and standard')

    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='exp1', help='folder name results are stored into')
    args = parser.parse_args()


    # training styles: 'independent', 'sequential', 'gan'
    # main('independent', 'exp1')
    # main('sequential', 'exp2')
    main('exp34_geometric', geometric=True, delta=True)