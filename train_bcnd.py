import torch
from models import GaussianPolicy
from utils import MyData
from torch.utils.data import DataLoader
import numpy as np
import argparse
import pickle


# train bcnd model
def train_model(args):

    # training parameters
    print("[-] training bcnd")
    EPOCH = 1000
    LR = 0.001

    # split the dataset into K disjoint sets
    D = np.array(pickle.load(open(args.loadname, "rb")))
    np.random.shuffle(D)
    datasets = np.split(D, args.K)

    # for each iteration M
    for m_count in range(args.M):

        print("training iteration m =", m_count)

        # load the previous models (if applicable)
        if m_count:
            prev_models = []
            for k_count in range(args.K):
                prev_model = GaussianPolicy(state_dim=4, hidden_dim=64, action_dim=2).to(device=args.device)
                prev_model.load_state_dict(torch.load("models/bcnd" + str(k_count)), "weights_only=True")
                prev_models.append(prev_model.eval())

        # for each of the K models
        for k_count in range(args.K):

            print("training model k =", k_count)

            # initialize the k-th model and optimizer
            model = GaussianPolicy(state_dim=4, hidden_dim=64, action_dim=2).to(device=args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # initialize the k-th dataset
            train_data = MyData(datasets[k_count])
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

                    # get R(s, a) = pi_prev(a \mid s)
                    if m_count:
                        prev_prob = torch.zeros_like(log_pi)
                        for pi_prev in prev_models:
                            prev_log_pi = pi_prev.get_log_prob(states, actions).detach()
                            prev_log_pi = prev_log_pi.sum(dim=1)
                            prev_prob += torch.nn.functional.softmax(prev_log_pi, dim=0) / args.K
                    # R = 1 for the first iteration
                    else:
                        prev_prob = torch.ones_like(log_pi)

                    # loss is -R(s, a) * log_pi
                    loss = torch.mean(-prev_prob * log_pi)
                         
                    # update model parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if epoch % 500 == 0:
                    print(epoch, loss.item())
                    torch.save(model.state_dict(), "models/bcnd" + str(k_count))

# train models
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadname', default="data/demos.pkl")
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--M', type=int, default=3)
    args = parser.parse_args()
    train_model(args)