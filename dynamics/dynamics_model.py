import torch
import torch.nn as nn

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
        x = torch.reshape(x, (x.size()[0], 64, 256))
        return x