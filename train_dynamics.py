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

# import utils as utils # TODO: add in utils file in dynamics folder and change input
# import emd.emd_module as emd
# import dynamics_model as dynamics # TODO: change import structure for dynamics model
# from action_dataset import ActionDataset # TODO: change import structure for dynamics dataset
from dynamics import dynamics_utils as utils
from dynamics import dynamics_model as dynamics
from dynamics.dynamics_dataset import DemoActionDataset

def get_dataloaders(pcl_type):
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

def train_dynamics(dvae, dynamics_network, optimizer, train_loader, epoch, device, recon_loss, combo_loss):
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

        if recon_loss:
            # TODO: NEED TO FIGURE OUT HOW TO GET THE CORRECT COMBO OF STUFF IN RET
                # NEED TO GET RET TO HAVE THE GT OF NEXT STATE NOT STATE!!!!!

            # reconstruct predicted next state
            ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, states) #.to(device)
            _, neighborhood_ns, _, _ = dvae.encode(next_states)
            ret = (ret_recon_next[0], ret_recon_next[1], ret_recon_next[2], ret_recon_next[3], neighborhood_ns, ret_recon_next[5])
            # orig_next_states = torch.reshape(orig_next_states.to(torch.float32), (recon_next_states.size())).to(device)
            # loss  = chamfer_distance(orig_next_states, recon_next_states)[0]
            loss = dvae.recon_loss(ret, next_states)
        elif combo_loss:
            z_next_states, neighborhood_ns, _, _ = dvae.encode(next_states) #.to(device)
            mse_func = nn.MSELoss()
            mse = mse_func(z_next_states, z_pred_next_states)
            # recon_next_states = dvae.decode(z_pred_next_states).to(device)
            # orig_next_states = torch.reshape(orig_next_states.to(torch.float32), (recon_next_states.size())).to(device)
            # cd  = chamfer_distance(orig_next_states, recon_next_states)[0]
            # ret = dvae.decode(z_pred_next_states, neighborhood, center, logits, states).to(device)
            # cd = dvae.recon_loss(ret, states)
            ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, states) #.to(device)
            ret = (ret_recon_next[0], ret_recon_next[1], ret_recon_next[2], ret_recon_next[3], neighborhood_ns, ret_recon_next[5])
            cd = dvae.recon_loss(ret, next_states)
            loss = mse + cd
        else:
            # get ground truth latent next state
            z_next_states, _, _, _ = dvae.encode(next_states) #.to(device)
            loss_func = nn.MSELoss()
            loss = loss_func(z_next_states, z_pred_next_states)

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

def test_dynamics(dvae, dynamics_network, optimizer, test_loader, epoch, device, recon_loss, combo_loss):
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

            if recon_loss:
                # reconstruct predicted next state
                ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, states) #.to(device)
                _, neighborhood_ns, _, _ = dvae.encode(next_states)
                ret = (ret_recon_next[0], ret_recon_next[1], ret_recon_next[2], ret_recon_next[3], neighborhood_ns, ret_recon_next[5])
                loss = dvae.recon_loss(ret, next_states)
            elif combo_loss:
                z_next_states, neighborhood_ns, _, _ = dvae.encode(next_states) #.to(device)
                mse_func = nn.MSELoss()
                mse = mse_func(z_next_states, z_pred_next_states)
                ret_recon_next = dvae.decode(z_pred_next_states, neighborhood, center, logits, states) #.to(device)
                ret = (ret_recon_next[0], ret_recon_next[1], ret_recon_next[2], ret_recon_next[3], neighborhood_ns, ret_recon_next[5])
                cd = dvae.recon_loss(ret, next_states)
                loss = mse + cd
            else:
                # get ground truth latent next state
                z_next_states, _, _, _ = dvae.encode(next_states) #.to(device)
                loss_func = nn.MSELoss()
                loss = loss_func(z_next_states, z_pred_next_states)

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
    # with open(join(folder_name, 'params.json'), 'w') as f:
    #     json.dump(save_args, f)

    device = torch.device('cuda')

    # params_file = open('out/' + args.load_path + '/params.json', 'r')
    # params = json.load(params_file)

    # z_dim = params['z_dim']
    z_dim = 64*256
    input_dim = args.enc_dim*256
    dynamics_network = dynamics.LatentVAEDynamicsNet(input_dim, z_dim, args.a_dim).to(device)
    parameters = list(dynamics_network.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = MultiStepLR(optimizer,
    #                 milestones=[25, 50, 100, 150, 200, 300],
    #                 # milestones=[75, 150, 200, 300],
    #                 gamma=0.25)

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

    train_loader, test_loader = get_dataloaders(args.pcl_type)

    best_test_loss = float('inf')
    itr = 0
    for epoch in range(args.epochs):
        # Train
        stats = train_dynamics(dvae, dynamics_network, optimizer, train_loader, epoch, device, args.recon_loss, args.combo_loss)
        test_loss = test_dynamics(dvae, dynamics_network, optimizer, test_loader, epoch, device, args.recon_loss, args.combo_loss)

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
    parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, help='default 0')
    parser.add_argument('--epochs', type=int, default=220, help='default: 100')
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    # parser.add_argument('--load_checkpoint', type=bool, default=True, help='if True, we are finetuning')
    parser.add_argument('--batch_size', type=int, default=32, help='default 32')
    parser.add_argument('--a_dim', type=int, default=5, help='dimension of the action')
    parser.add_argument('--enc_dim', type=int, default=8, help='dimension of the action')
    parser.add_argument('--combo_loss', type=bool, default=False, help='scaled combo of MSE and CD')
    parser.add_argument('--recon_loss', type=bool, default=False, help='loss in reconstructed space, or latent space')
    parser.add_argument('--n_pts', type=int, default=2048, help='number of points in point cloud') 
    parser.add_argument('--pcl_type', type=str, default='shell_scaled', help='options: dense_centered, dense_scaled, shell_centered, shell_scaled')
                                                                                    
    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='exp2_twonetworks_mse', help='folder name results are stored into')
    args = parser.parse_args()

    main()

