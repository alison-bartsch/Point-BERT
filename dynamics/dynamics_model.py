import torch
import torch.nn as nn

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