import torch
from models import GaussianPolicy
from utils import MyData
from torch.utils.data import DataLoader
import argparse


# train bc model
def train_model(args):

    # training parameters
    print("[-] training bc")
    EPOCH = 1000
    LR = 0.001

    # initialize model and optimizer
    model = GaussianPolicy(state_dim=4, hidden_dim=64, action_dim=2).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # initialize dataset
    print("[-] loading data: " + args.loadname)
    train_data = MyData(None, args.loadname)
    BATCH_SIZE = int(len(train_data) / 10.)
    print("my batch size is:", BATCH_SIZE)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # main training loop
    for epoch in range(EPOCH+1):
        for batch, x in enumerate(train_set):
        
            # start with standard bc
            states = x[:, 0, 0:4].to(device=args.device)
            actions = x[:, 0, 4:6].to(device=args.device)
            log_pi = model.get_log_prob(states, actions)
            log_pi = log_pi.sum(dim=1)
            
            # loss is -log_pi
            loss = torch.mean(-log_pi)
                 
            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 500 == 0:
            print(epoch, loss.item())
            torch.save(model.state_dict(), "models/bc")

# train models
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadname', default="data/demos.pkl")
    parser.add_argument('--device', default="cpu")
    args = parser.parse_args()
    train_model(args)
