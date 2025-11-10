# Counter-BC
Implementation of the Counter-BC algorithm from the paper "Counterfactual behavior cloning: Offline imitation learning from imperfect human demonstrations"

## Installation and Setup

To install Counter-BC, clone the repo using:
```
git clone https://github.com/VT-Collab/Counter_BC.git
```

You will need to have numpy and torch installed:
```
pip3 install numpy torch
```

## Contents

This respository contains a minimal implementation of the Counter-BC algorithm and baselines, as well as the scripts needed to reproduce our results from the "Intersection" environment. You can run the entire data collection, training, and testing pipeline using:
```
./bash.sh
```
A description of each file is provided below:
- `models.py`: Gaussian policy. Trained instances of this policy are saved in "models"
- `train_bc.py`: Trains the Gaussian policy using standard behavior cloning conditioned on the current state
- `train_bc-rnn.py`: Trains the GaussianRNN policy using standard behavior cloning conditioned on a history of states
- `train_ileed.py`: Trains the Gaussian policy using a modified version of ILEED ("Imitation Learning by Estimating Expertise of Demonstrators")
- `train_bcnd.py`: Trains the Gaussian policy using BCND method by Sasaki and Yamashina ("Behavioral Cloning from Noisy Demonstrations")
- `train_counter-bc.py`: Our proposed Algorithm 1 in the manuscript
- `get_data.py`: Collects a dataset of synthetic demonstrations in the Intersection environment, and saves them to the folder "data"
- `test.py`: Rolls out the trained policies in the Intersection environment and saves their performance to the folder "results"
