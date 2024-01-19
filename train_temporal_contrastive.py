import torch
import torch.nn as nn
import torchvision
import os
import json
from tools import builder

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from geometric_utils import *
from utils.config import cfg_from_yaml_file

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
# from pytorch3d.loss import chamfer_distance # TODO: perhaps use this repository's version of CD????
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from dynamics import dynamics_utils as utils
from dynamics import dynamics_model as dynamics
from dynamics.dynamics_dataset import DemoActionDataset, GeometricDataset, FeatureDynamicsDataset, DemoContrastiveDataset

def get_dataloaders(pcl_type):
    """
    Insert comment
    """
    full_dataset = DemoContrastiveDataset('/home/alison/Clay_Data/Fully_Processed/Jan17_Human_Demos', pcl_type)
    # full_dataset = DemoActionDataset('/home/alison/Clay_Data/Fully_Processed/Aug29_Correct_Scaling_Human_Demos', pcl_type)
    # full_dataset = DemoActionDataset('home/alison/Clay_Data/Fully_Processed/Sept11_Random', pcl_type)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size])
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# def info_nce_loss(states, next_states):
#     # define temperature
#     temp = 1

#     # 

def compute_cpc_loss(obs, obs_pos, obs_neg):
    """
    Insert comment
    """
    bs = obs.shape[0]

    # already get the latent encoding of the states, next states (positive samples) and random (negative samples)
    # get the negative dot product between the state and random samples
    neg_dot_products = torch.mm(obs, obs_neg.t()) # b x b

    # get the sum of similarity function on the state and random samples
    neg_dists = -((obs ** 2).sum(1).unsqueeze(1) - 2* neg_dot_products + (obs_neg ** 2).sum(1).unsqueeze(0))
    idxs = np.arange(bs)
    # Set to minus infinity entries when comparing z with z - will be zero when apply softmax
    neg_dists[idxs, idxs] = float('-inf') # b x b+1

    # get the similarity function on the state and next state (positive samples)
    pos_dot_products = (obs_pos * obs).sum(dim=1) # b
    pos_dists = -((obs_pos ** 2).sum(1) - 2* pos_dot_products + (obs ** 2).sum(1))
    pos_dists = pos_dists.unsqueeze(1) # b x 1

    # get the log
    dists = torch.cat((neg_dists, pos_dists), dim=1) # b x b+1
    dists = F.log_softmax(dists, dim=1) # b x b+1
    loss = -dists[:, -1].mean() # Get last column with is the true pos sample

    return loss

    # latent encoding of the observations and the positive observations
    # z, z_pos = encoder(obs.to(torch.float32)), encoder(obs_pos.to(torch.float32))  # b x z_dim

    # predict the next latent observation given actions
    # z_next = trans(z, actions)

    # get negative dot products between the predicted next and any wrong encoded state transposed
    # neg_dot_products = torch.mm(z_next, z.t()) # b x b

    # get the sum of the similarity function on the predicted next and wrong encoded states
    # neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2* neg_dot_products + (z ** 2).sum(1).unsqueeze(0))
    # idxs = np.arange(bs)
    # # Set to minus infinity entries when comparing z with z - will be zero when apply softmax
    # neg_dists[idxs, idxs] = float('-inf') # b x b+1

    # get the similarity function on the predicted next and ground truth encoded states
    # pos_dot_products = (z_pos * z_next).sum(dim=1) # b
    # pos_dists = -((z_pos ** 2).sum(1) - 2* pos_dot_products + (z_next ** 2).sum(1))
    # pos_dists = pos_dists.unsqueeze(1) # b x 1

    # get the log
    # dists = torch.cat((neg_dists, pos_dists), dim=1) # b x b+1
    # dists = F.log_softmax(dists, dim=1) # b x b+1
    # loss = -dists[:, -1].mean() # Get last column with is the true pos sample

    # return loss

def train_action(pointbert, encoder_head, optimizer, scheduler, train_loader, epoch):
    # pointbert.eval()
    pointbert.train()
    encoder_head.train()

    stats = utils.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    parameters = list(encoder_head.parameters()) 

    for states, next_states, rand_states in train_loader:

        states = states.cuda()
        next_states = next_states.cuda()
        rand_states = rand_states.cuda()

        tokenized_states = pointbert(states)
        tokenized_next_states = pointbert(next_states)
        tokenized_rand_states = pointbert(rand_states)

        latent_states = encoder_head(tokenized_states)
        latent_next_states = encoder_head(tokenized_next_states)
        latent_rand_states = encoder_head(tokenized_rand_states)

        # have the contrastive loss here!
        loss = compute_cpc_loss(latent_states, latent_next_states, latent_rand_states)

        # loss_func = nn.CrossEntropyLoss()
        # loss = loss_func(info_nce_loss(latent_states, latent_next_states))

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

def test_action(pointbert, encoder_head, optimizer, test_loader, epoch):
    pointbert.eval()
    encoder_head.eval()

    test_loss = 0
    for states, next_states, rand_states in test_loader:
        with torch.no_grad():
            states = states.cuda()
            next_states = next_states.cuda()
            rand_states = rand_states.cuda()

            tokenized_states = pointbert(states)
            tokenized_next_states = pointbert(next_states)
            tokenized_rand_states = pointbert(rand_states)

            latent_states = encoder_head(tokenized_states)
            latent_next_states = encoder_head(tokenized_next_states)
            latent_rand_states = encoder_head(tokenized_rand_states)

            # have the contrastive loss here!
            loss = compute_cpc_loss(latent_states, latent_next_states, latent_rand_states)

            # loss_func = nn.MSELoss()
            # loss = loss_func(pred_actions, actions)

        test_loss += loss * states.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Feat Test Loss: {test_loss:.4f}')
    return test_loss.item()

def main(exp_name, geometric=True, delta=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('embedding_experiments', exp_name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    writer = SummaryWriter(join(folder_name, 'data'))

    # learning rate scheduler parameters
    milestones = [10,25,50,75,100,125,150,175,200,225,250,300,350,450] # [10,25,50,75,100,150,200,250,300,350,450]
    gamma = 0.5 # 0.25

    save_args = vars(args)
    save_args['milestones'] = milestones
    save_args['gamma'] = gamma
    save_args['script'] = 'train_action'
    with open(join(folder_name, 'params.json'), 'w') as f:
        json.dump(save_args, f) 

    device = torch.device('cuda')

    # load the encoder head and action network
    encoded_dim = 768 # 768 # 65*384
    latent_dim = 512
    encoder_head = dynamics.EncoderHead(encoded_dim, latent_dim).to(device)

    parameters = list(encoder_head.parameters())
   
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                    # milestones=[15, 45, 75, 100, 200],
                    # milestones=[10,25,50,75,100,125,150,200,250,300,350,450],
                    # gamma=0.5)
                    # milestones=[25,50,75,100,125,150,200,250,300,350,450],
                    # gamma=0.25)
                    milestones=milestones,
                    gamma=gamma)

    config = cfg_from_yaml_file('cfgs/ModelNet_models/PointTransformer.yaml')
    model_config = config.model
    pointbert = builder.model_builder(model_config)
    weights_path = 'experiments/Point-BERT/Mixup_models/downloaded/Point-BERT.pth'
    pointbert.load_model_from_ckpt(weights_path)
    pointbert.to(device)

    train_loader, test_loader = get_dataloaders(args.pcl_type)

    print('--------------------------- Training Action Network ---------------------------')
    best_test_loss = float('inf')
    itr = 0
    for epoch in range(args.epochs):

        stats = train_action(pointbert, encoder_head, optimizer, scheduler, train_loader, epoch)
        test_loss = test_action(pointbert, encoder_head, optimizer, test_loader, epoch)

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

                # save current pointbert model!
                # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                torch.save({
                    'base_model' : pointbert.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : dict(),
                    'best_metrics' : dict(),
                    }, os.path.join(folder_name, 'best_pointbert.pth'))
                
                checkpoint = {
                    'encoder_head': encoder_head, 
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
    parser.add_argument('--epochs', type=int, default=250, help='default: 100') # 500
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--batch_size', type=int, default=32, help='default 32') # 32

    # Action and Cloud Parameters
    parser.add_argument('--a_dim', type=int, default=5, help='dimension of the action')
    parser.add_argument('--n_pts', type=int, default=2048, help='number of points in point cloud') 
    parser.add_argument('--pcl_type', type=str, default='shell_scaled', help='options: dense_centered, dense_scaled, shell_centered, shell_scaled')
 
    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='exp1', help='folder name results are stored into')
    args = parser.parse_args()

    main('exp1_statenextstate_contrastive')


# # setup the Point-BERT model loading the checkpoint
# # TODO: convert this dictionary into .yaml file
# config_transformer = {'NAME': 'PointTransformer', 
#                       'trans_dim': 384, 
#                       'depth': 12, 
#                       'drop_path_rate': 0.1, 
#                       'cls_dim': 40, 
#                       'num_heads': 6, 
#                       'group_size': 32, 
#                       'num_group': 64, 
#                       'encoder_dims': 256}
# base_model = builder.model_builder(config_transformer)
# base_model.load_model_from_ckpt('experiments/Point-BERT/Mixup_models/downdloaded/Point-BERT.pth')

# # pass the batch of point clouds through Point-BERT to the get the embeddings
# # pass the embeddings through the action network head to get the predicted action