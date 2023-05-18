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

# import utils as utils # TODO: add in utils file in dynamics folder and change input
# import emd.emd_module as emd
# import dynamics_model as dynamics # TODO: change import structure for dynamics model
# from action_dataset import ActionDataset # TODO: change import structure for dynamics dataset
from dynamics import dynamics_utils as utils
from dynamics import dynamics_model as dynamics
from dynamics.dynamics_dataset import DemoActionDataset, DemoWordDataset

def get_dataloaders(pcl_type, word_dynamics, dvae=None):
    """
    Insert comment
    """
    # full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/May10_5D', pcl_type)
    if word_dynamics:
        full_dataset = DemoWordDataset('/home/alison/Clay_Data/Fully_Processed/All_Shapes', pcl_type, dvae)
    else:
        full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/All_Shapes', pcl_type)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size])
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_word_dynamics(dvae, dynamics_network, optimizer, scheduler, train_loader, epoch, device, loss_type):
    dvae.eval()
    dynamics_network.train()

    stats = utils.Stats()
    pbar = tqdm(total=len(train_loader.dataset)*64)
    parameters = list(dynamics_network.parameters())

    for states, next_states, actions, word_idx in train_loader:
        states = states.cuda()
        next_states = states.cuda()
        actions = actions.cuda()

        z_states, neighborhood, centers, _ = dvae.encode(states) 
        # gt_z_next_states, _, _, _ = dvae.encode(next_states)
        gt_z_next_states, _, _, _ = dvae.next_state_encode(next_states, centers, neighborhood)

        if states.size()[0] > 1:
            batch_idxs = torch.linspace(0,states.size()[0]-1, steps=states.size()[0], dtype=int)
            ns_word = gt_z_next_states[batch_idxs,word_idx,:] # NOTE: convert target to one-hot by looking up in codebook
            target = dvae.codebook_onehot(ns_word).cuda()
            group_center = centers[batch_idxs,word_idx,:]
            vocab = z_states[batch_idxs,word_idx,:] # 32, 256
        else: # NOTE: SO MUCH FASTER!
            ns_word = gt_z_next_states.squeeze()
            target = dvae.codebook_onehot(ns_word).cuda()
            group_center = centers.squeeze()
            vocab = z_states.squeeze()
            actions = torch.tile(actions, (vocab.size()[0], 1))

        ns_logits = dynamics_network(vocab, group_center, actions) #.to(device)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(ns_logits, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}')
        pbar.update(vocab.shape[0])
    pbar.close()
    scheduler.step()
    return stats

def test_word_dynamics(dvae, dynamics_network, optimizer, test_loader, epoch, device, loss_type):
    dvae.eval()
    dynamics_network.eval()

    test_loss = 0
    for states, next_states, actions, word_idx in test_loader:
        with torch.no_grad():
            states = states.cuda()
            next_states = states.cuda()
            actions = actions.cuda()

            z_states, neighborhood, centers, _ = dvae.encode(states) 
            # gt_z_next_states, _, _, _ = dvae.encode(next_states)
            gt_z_next_states, _, _, _ = dvae.next_state_encode(next_states, centers, neighborhood)

            if states.size()[0] > 1:
                batch_idxs = torch.linspace(0,states.size()[0]-1, steps=states.size()[0], dtype=int)
                ns_word = gt_z_next_states[batch_idxs,word_idx,:] # NOTE: convert target to one-hot by looking up in codebook
                target = dvae.codebook_onehot(ns_word).cuda()
                group_center = centers[batch_idxs,word_idx,:]
                vocab = z_states[batch_idxs,word_idx,:] # 32, 256
            else:
                ns_word = gt_z_next_states.squeeze()
                target = dvae.codebook_onehot(ns_word).cuda()
                group_center = centers.squeeze()
                vocab = z_states.squeeze()
                actions = torch.tile(actions, (vocab.size()[0], 1))

            ns_logits = dynamics_network(vocab, group_center, actions) #.to(device)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(ns_logits, target)

            test_loss += loss * vocab.shape[0]
    test_loss /= (len(test_loader.dataset)*64)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    return test_loss.item()

def train_center_dynamics(dvae, dynamics_network, optimizer, train_loader, epoch, device, loss_type):
    dvae.eval()
    dynamics_network.train()

    stats = utils.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    parameters = list(dynamics_network.parameters())
    for states, next_states, actions in train_loader:

        states = states.cuda()
        next_states = states.cuda()
        actions = actions.cuda()

        _, _, center_state, _ = dvae.encode(states) #.to(device)
        _, _, center_next_states, _ = dvae.encode(next_states)
        # z_pred_next_states = dynamics_network(z_states, actions).to(device)

        ns_center_pred = dynamics_network(center_state, actions).to(device)
        # loss_func = nn.MSELoss()
        # loss = loss_func(center_next_states, ns_center_pred)
        loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}')
        pbar.update(states.shape[0])
    pbar.close()
    return stats

def test_center_dynamics(dvae, dynamics_network, optimizer, test_loader, epoch, device, loss_type):
    dvae.eval()
    dynamics_network.eval()

    test_loss = 0
    for states, next_states, actions in test_loader:

        states = states.cuda()
        next_states = states.cuda()
        actions = actions.cuda()

        _, _, center_state, _ = dvae.encode(states) #.to(device)
        _, _, center_next_states, _ = dvae.encode(next_states)
        # z_pred_next_states = dynamics_network(z_states, actions).to(device)

        ns_center_pred = dynamics_network(center_state, actions).to(device)

        # try mse loss
        # loss_func = nn.MSELoss()
        # loss = loss_func(center_next_states, ns_center_pred)
        loss = chamfer_distance(center_next_states, ns_center_pred)[0]

        # could do chamfer distance

        test_loss += loss * states.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    return test_loss.item()

def train_dynamics(dvae, dynamics_network, optimizer, train_loader, epoch, device, loss_type):
    dvae.eval()
    dynamics_network.train()

    stats = utils.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    parameters = list(dynamics_network.parameters())
    for states, next_states, actions in train_loader:

        states = states.cuda()
        next_states = states.cuda()
        actions = actions.cuda()

        z_states, neighborhood, center, logits = dvae.encode(states) #.to(device)
        z_pred_next_states = dynamics_network(z_states, actions).to(device)

        if loss_type == 'cd':
            # reconstruct predicted next state
            ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, states) #.to(device)
            _, neighborhood_ns, _, _ = dvae.encode(next_states)
            ret = (ret_recon_next[0], ret_recon_next[1], ret_recon_next[2], ret_recon_next[3], neighborhood_ns, ret_recon_next[5])
            loss = dvae.recon_loss(ret, next_states)
        elif loss_type == 'both':
            z_next_states, neighborhood_ns, _, _ = dvae.encode(next_states) #.to(device)
            mse_func = nn.MSELoss()
            mse = mse_func(z_next_states, z_pred_next_states)
            ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, states) #.to(device)
            ret = (ret_recon_next[0], ret_recon_next[1], ret_recon_next[2], ret_recon_next[3], neighborhood_ns, ret_recon_next[5])
            cd = dvae.recon_loss(ret, next_states)
            loss = 100*mse + cd
        elif loss_type == 'mse':
            # get ground truth latent next state
            z_next_states, _, _, _ = dvae.encode(next_states) #.to(device)
            loss_func = nn.MSELoss()
            loss = loss_func(z_next_states, z_pred_next_states)
        elif loss_type == 'cos':
            z_next_states, _, _, _ = dvae.encode(next_states) #.to(device)
            loss_func = nn.CosineSimilarity(dim=2) # previously dim=1, but corresponding to patches not words!
            word_losses = 1 - loss_func(z_next_states, z_pred_next_states)
            loss = torch.sum(word_losses, axis=1)
            loss = torch.mean(loss) # range: 0 -> 512

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}')
        pbar.update(states.shape[0])
    pbar.close()
    return stats

def test_dynamics(dvae, dynamics_network, optimizer, test_loader, epoch, device, loss_type, word_dynamics):
    dvae.eval()
    dynamics_network.eval()

    test_loss = 0
    for states, next_states, actions in test_loader:
        with torch.no_grad():
            states = states.cuda()
            next_states = states.cuda()
            actions = actions.cuda()

            z_states, neighborhood, center, logits = dvae.encode(states) #.to(device)
            z_pred_next_states = dynamics_network(z_states, actions).to(device)

            if loss_type == 'cd':
                # reconstruct predicted next state
                ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, states) #.to(device)
                _, neighborhood_ns, _, _ = dvae.encode(next_states)
                ret = (ret_recon_next[0], ret_recon_next[1], ret_recon_next[2], ret_recon_next[3], neighborhood_ns, ret_recon_next[5])
                loss = dvae.recon_loss(ret, next_states)
            elif loss_type == 'both':
                z_next_states, neighborhood_ns, _, _ = dvae.encode(next_states) #.to(device)
                mse_func = nn.MSELoss()
                mse = mse_func(z_next_states, z_pred_next_states)
                ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, states) #.to(device)
                ret = (ret_recon_next[0], ret_recon_next[1], ret_recon_next[2], ret_recon_next[3], neighborhood_ns, ret_recon_next[5])
                cd = dvae.recon_loss(ret, next_states)
                loss = 100*mse + cd
            elif loss_type == 'mse':
                # get ground truth latent next state
                z_next_states, _, _, _ = dvae.encode(next_states) #.to(device)
                loss_func = nn.MSELoss()
                loss = loss_func(z_next_states, z_pred_next_states)
            elif loss_type == 'cos':
                z_next_states, _, _, _ = dvae.encode(next_states) #.to(device)
                loss_func = nn.CosineSimilarity(dim=2)
                word_losses = 1 - loss_func(z_next_states, z_pred_next_states)
                loss = torch.sum(word_losses, axis=1)
                loss = torch.mean(loss)

            test_loss += loss * states.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    return test_loss.item()

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('dvae_dynamics_experiments', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    writer = SummaryWriter(join(folder_name, 'data'))

    save_args = vars(args)
    save_args['script'] = 'train_dynamics'
    with open(join(folder_name, 'params.json'), 'w') as f:
        json.dump(save_args, f)

    device = torch.device('cuda')

    # params_file = open('out/' + args.load_path + '/params.json', 'r')
    # params = json.load(params_file)

    # z_dim = params['z_dim']
    
    if args.word_dynamics:
        input_dim = 256 + args.a_dim + 3 
        output_dim = 8192
        dynamics_network = dynamics.NeighborhoodDynamics(input_dim, output_dim).to(device)
    elif args.center_dynamics:
        dim = 64*3
        input_dim = dim + args.a_dim
        # dynamics_network = dynamics.CenterDynamics(input_dim, dim).to(device)
        dynamics_network = dynamics.PointNetDynamics(dim).to(device)
    else:
        z_dim = 64*256 # 8192 # 256
        input_dim = args.enc_dim*256
        dynamics_network = dynamics.LatentVAEDynamicsNet(input_dim, z_dim, args.a_dim).to(device)
    parameters = list(dynamics_network.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                    milestones=[25, 50, 100, 150, 200, 300],
                    gamma=0.25)

    # load_name = join('out', args.load_path)
    # checkpoint = torch.load(join(load_name, 'checkpoint'))
    # encoder = checkpoint['encoder'].to(device)
    # decoder = checkpoint['decoder'].to(device)

    # load the pre-trained dvae
    config = cfg_from_yaml_file('cfgs/Dynamics/dvae.yaml')
    config=config.config
    model_path = 'experiments/dvae/ShapeNet55_models/test_dvae/ckpt-best.pth' # set correct model path to load from
    dvae = builder.model_builder(config)
    builder.load_model(dvae, model_path, logger = 'dvae_testclay')
    dvae.to(device)
    # dvae.to(0)

    train_loader, test_loader = get_dataloaders(args.pcl_type, args.word_dynamics, dvae)

    best_test_loss = float('inf')
    itr = 0
    for epoch in range(args.epochs):
        # Train
        if args.word_dynamics:
            stats = train_word_dynamics(dvae, dynamics_network, optimizer, train_loader, epoch, device, args.loss_type)
            test_loss = test_word_dynamics(dvae, dynamics_network, optimizer, test_loader, epoch, device, args.loss_type)
        elif args.center_dynamics:
            stats = train_center_dynamics(dvae, dynamics_network, optimizer, train_loader, epoch, device, args.loss_type)
            test_loss = test_center_dynamics(dvae, dynamics_network, optimizer, test_loader, epoch, device, args.loss_type)
        else:
            stats = train_dynamics(dvae, dynamics_network, optimizer, train_loader, epoch, device, args.loss_type)
            test_loss = test_dynamics(dvae, dynamics_network, optimizer, test_loader, epoch, device, args.loss_type)

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

                with open(folder_name + '/best_test_loss.txt', 'w') as f:
                    f.write(str(test_loss))

                checkpoint = {
                    'dynamics_network': dynamics_network,
                    'optimizer': optimizer,
                }

                torch.save(checkpoint, join(folder_name, 'checkpoint'))
                print('Saved models with loss', best_test_loss)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, help='default 0')
    parser.add_argument('--epochs', type=int, default=500, help='default: 100')
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--batch_size', type=int, default=32, help='default 32')

    # Action and Cloud Parameters
    parser.add_argument('--a_dim', type=int, default=5, help='dimension of the action')
    parser.add_argument('--enc_dim', type=int, default=16, help='dimension of the action')
    parser.add_argument('--loss_type', type=str, default='mse', help='[cos, mse, cd, both]')
    parser.add_argument('--n_pts', type=int, default=2048, help='number of points in point cloud') 
    parser.add_argument('--pcl_type', type=str, default='shell_scaled', help='options: dense_centered, dense_scaled, shell_centered, shell_scaled')
    parser.add_argument('--word_dynamics', type=bool, default=False, help='dynamics model at word-level or global')                                                                                
    parser.add_argument('--center_dynamics',type=bool, default=True, help='dynamics model for the centroids' )

    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='exp12_center_pointnet', help='folder name results are stored into')
    args = parser.parse_args()

    main()

