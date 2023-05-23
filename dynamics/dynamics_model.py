import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dvae import *

class CGCNNDynamics(nn.Module):
    def __init__(self, action_dims, token_dims, decoder_dims, n_tokens):
        super(CGCNNDynamics, self).__init__()
        self.n_tokens = n_tokens
        self.action_dims = action_dims
        self.token_dims = token_dims
        self.decoder_dims = decoder_dims
        self.input_dims = self.action_dims + self.token_dims
        self.dgcnn = DGCNN(encoder_channel = self.input_dims, output_channel = self.decoder_dims)
    
    def forward(self, sampled, center, action):
        action = torch.tile(torch.unsqueeze(action, 1), (1, self.n_tokens, 1))
        inp = torch.concat((sampled, action), dim=2)
        feature = self.dgcnn(inp, center)
        return feature
        

class PointNetfeatModified(nn.Module):
    def __init__(self, global_feat = True):
        super(PointNetfeatModified, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.global_feat = global_feat

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 512, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)
        
class PointNetDynamics(nn.Module):
    def __init__(self, output_dim):
        super(PointNetDynamics, self).__init__()
        self.output_dim = output_dim

        self.feat = PointNetfeatModified(global_feat=True)
        self.fc1 = nn.Linear(512 + 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.output_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        # self.relu = nn.ReLU()

    def forward(self, centers, action):
        size = centers.size()
        centers = torch.reshape(centers, (size[0], size[2], size[1]))
        x = self.feat(centers)
        x = torch.cat((x, action), dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = torch.reshape(x, (x.size()[0], 64, 3))
        return x

class CenterDynamics(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim # 64 centroids x 3 (x,y,z)
        self.output_dim = output_dim
        self.hidden_size = 512

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.hidden_size, self.output_dim)
        )
    
    def forward(self, center, action):
        center = torch.reshape(center, (center.size()[0], self.output_dim)) # reshape to (batch_size, self.dim)
        x = torch.cat((center, action), dim=-1)
        x = self.model(x)
        x = torch.reshape(x, (x.size()[0], 64, 3))
        return x

class NeighborhoodDynamics(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim # 256 + 5 + 3 = 264
        self.output_dim = output_dim # logits 8192
        self.hidden_size = 512

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2*self.hidden_size, 4*self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4*self.hidden_size, self.output_dim)
        )

        # self.model = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(self.hidden_size, self.output_dim)
        # )

    def forward(self, vocab, center, action):
        # given the neighborhood, center, action 

        """
        1) given neighborhood, center, action and current vocab, predict new vocab, and neighborhood
        2) model A predicts new vocab and model B predicts new neighborhood
            # A: given current vocab, action and center, generate logits and sample new vocab
                # train as a classifier 
            # B: given current neighborhood, center, action and new vocab generate new neighborhood?
                # train with ground truth new vocab and re-calculated neighborhood from stagnant center
        """
        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # f = r-a  # free inside reserved
        # # print("\nFree GPU Space: ", f)
        # print("\nTotal Memory: ", t)
        # print("reserved" , r)
        # print("allocated: ", a)
        x = torch.cat((vocab, center, action), dim=-1)
        out = self.model(x)
        return out
        

class SharedEncoder(nn.Module):
    """
    Simple encoder to encode each of the vocabulary of the dvae to a lower
    dimensional space
    """
    def __init__(self, input_dim):
        super().__init__()

        hidden_size = 64
        latent_size = int(input_dim/256)

        self.model = nn.Sequential(
            nn.Linear(64, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, int(0.5*hidden_size)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(0.5*hidden_size), latent_size) # switch to 8?
        )

    def forward(self, z):
        z = torch.reshape(z, (z.size()[0], z.size()[2], z.size()[1]))
        x = self.model(z)
        return x
    
class LatentVAEDynamicsNet(nn.Module):
    def __init__(self, input_dim, z_dim, action_dim):
        super().__init__()

        self.vocab_encoder = SharedEncoder(input_dim)
        hidden_size = 2048
        self.model = nn.Sequential(
            nn.Linear(input_dim + action_dim, 4*hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4*hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2*hidden_size, 4*hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4*hidden_size, z_dim)
        )

    def forward(self, z, a):
        x = self.vocab_encoder(z)
        x = torch.reshape(x, (x.size()[0], x.size()[1]*x.size()[2]))

        # reshape x to be a single dimension
        x = torch.cat((x, a), dim=-1)
        x = self.model(x)
        x = torch.reshape(x, (x.size()[0], 64, 256)) # 8192
        # x = torch.reshape(x, (x.size()[0], 64, 8192)) 
        return x