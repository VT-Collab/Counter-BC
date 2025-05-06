import torch
import numpy as np
from models import GaussianPolicy
from get_data import agent
import argparse
import csv


# rollout using custom intersection environment
# this environment matches the environment used during data collection
def rollout(start_state, model):
    total_cost = 0.
    state = np.copy(start_state)
    goal_x = np.array([10., 0.])
    goal_y = np.array([0., 10.])
    for idx in range(20):
        u1 = model(torch.FloatTensor(state)).detach().numpy()
        if np.linalg.norm(u1) > 1.0:
            u1 /= np.linalg.norm(u1)
        u2 = agent(state[2:4], state[0:2], goal_y)
        # compute cost; same cost used during data collection
        x, y = state[0:2], state[2:4]
        C_goal = np.linalg.norm((x + u1) - goal_x) - np.linalg.norm(x - goal_x)
        C_avoid = np.linalg.norm(x - y) - np.linalg.norm((x + u1) - y)
        total_cost += C_goal + 0.75 * C_avoid
        state[0:2] += u1
        state[2:4] += u2
    return -total_cost


# load and evaluate the models
def evaluate_model(loadname, start_states, args):
    model = GaussianPolicy(state_dim=4, hidden_dim=64, action_dim=2)
    model.load_state_dict(torch.load('models/' + loadname))
    avg_reward = 0.
    for s in start_states:
        reward = rollout(s, model)
        avg_reward += reward / len(start_states)
    print("[+] " + loadname + " average reward:", avg_reward)
    return avg_reward


# test trained models
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--savename', default="results.csv")
    args = parser.parse_args()

    # sample start states
    start_states = []
    for _ in range(100):
        start_x = np.random.uniform([-10, -10], [0, 10], 2)
        start_y = np.random.uniform([-10, -10], [10, 0], 2)
        start_state = np.array([start_x[0], start_x[1], start_y[0], start_y[1]])
        start_states.append(start_state)

    # rollout each trained policy over the start states
    test_names = ['bc', 'ileed', 'sasaki', 'counter-bc']
    score = []
    for loadname in test_names:
        score.append(evaluate_model(loadname, start_states, args))

    # write the results to a csv file
    with open("results/" + args.savename,'a') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(score)
