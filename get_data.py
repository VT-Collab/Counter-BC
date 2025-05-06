import pickle
import numpy as np
import argparse


# expert agent policy
# x is the position of the ego agent;
# y is the position of the other agent; 
# goal is the position of the ego agent's goal
# returns the sampled action that minimizes cost C
def agent(x, y, goal, beta=1.0):
    U = np.random.uniform(-1, 1, (100, 2))
    P = []
    for u in U:
        if np.linalg.norm(u) > 1.0:
            u /= np.linalg.norm(u)
        C_goal = np.linalg.norm((x + u) - goal) - np.linalg.norm(x - goal)
        C_avoid = np.linalg.norm(x - y) - np.linalg.norm((x + u) - y)
        C = C_goal + 0.75 * C_avoid
        P.append(np.exp(-beta * C))
    idx = np.argmax(P)
    return U[idx, :]


# get demonstrations
# each interaction consists of 20 timesteps
# during interactions two vehicles attempt to reach their goals
# while avoiding collision
def get_dataset(args):
    dataset = []
    goal_x = np.array([10., 0.])
    goal_y = np.array([0., 10.])
    for _ in range(args.n_interactions):
        # initial position of the two vehicles
        x = np.random.uniform([-10, -10], [0, 10], 2)
        y = np.random.uniform([-10, -10], [10, 0], 2)
        for idx in range(20):
            # optimal action for ego agent
            action = agent(x, y, goal_x)
            # add noise of the specified type
            if args.noise_type == "uniform":
                action += np.random.uniform(-args.sigma, +args.sigma, 2)
            elif args.noise_type == "gaussian":
                action += np.random.normal(0, args.sigma, 2)
            elif args.noise_type == "random":
                if np.random.rand() < args.sigma:
                    action += np.random.uniform(-args.sigma, +args.sigma, 2)
            # optimal action for the other agent
            u2 = agent(y, x, goal_y)
            state = np.array([x[0], x[1], y[0], y[1]])
            datapoint = []
            datapoint.append(list(state) + list(action))
            # sample counterfactuals within radius delta
            for _ in range(50):
                action1 = action + np.random.uniform(-args.delta, +args.delta, 2)
                datapoint.append(list(state) + list(action1))
            dataset.append(datapoint)
            x += action
            y += u2
    pickle.dump(dataset, open("data/demos.pkl", "wb"))
    print("dataset has this many state-action pairs:", len(dataset))


# save the specified number of demonstrations
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_interactions', type=int, default=10)
    parser.add_argument('--noise_type', default="uniform")
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.5)
    args = parser.parse_args()
    get_dataset(args)
