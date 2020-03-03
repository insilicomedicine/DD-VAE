import argparse
import gzip
import os
import pickle
import sys

import numpy as np
import rdkit
import scipy.stats as sps
import torch
from dd_vae.bo.sparse_gp import SparseGP
from dd_vae.bo.utils import max_ring_penalty
from dd_vae.utils import prepare_seed
from dd_vae.vae_rnn import VAE_RNN
from moses.metrics import SA
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles

rdkit.rdBase.DisableLog('rdApp.*')


# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest:
        dest.write(result)
    dest.close()


def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source:
        result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--load_dir", type=str, required=True)
args = parser.parse_args(sys.argv[1:])

prepare_seed(args.seed)

model = VAE_RNN.load(args.model).to(args.device)

# We load the data (y is minued!)
data = np.load(os.path.join(args.load_dir, 'features.npz'))
X = data['latent_points']
y = -data['targets']
y = y.reshape((-1, 1))

n = X.shape[0]

permutation = np.random.choice(n, n, replace=False)

X_train = X[permutation, :][0: np.int(np.round(0.9 * n)), :]
X_test = X[permutation, :][np.int(np.round(0.9 * n)):, :]

y_train = y[permutation][0: np.int(np.round(0.9 * n))]
y_test = y[permutation][np.int(np.round(0.9 * n)):]

np.random.seed(args.seed)

logP_values = data['logP_values']
SA_scores = data['SA_scores']
cycle_scores = data['cycle_scores']
SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(
    SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(
    logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(
    cycle_scores)) / np.std(cycle_scores)

iteration = 0
while iteration < 5:
    # We fit the GP
    np.random.seed(iteration * args.seed)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,
                       y_test, minibatch_size=10 * M, max_iterations=100,
                       learning_rate=0.001)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test) ** 2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale=np.sqrt(uncert)))
    print('Test RMSE:', error)
    print('Test ll:', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train) ** 2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale=np.sqrt(uncert)))
    print('Train RMSE:', error)
    print('Train ll:', trainll)

    # We pick the next 60 inputs
    iters = 60
    next_inputs = sgp.batched_greedy_ei(iters, np.min(X_train, 0),
                                        np.max(X_train, 0))
    valid_smiles = []
    new_features = []
    for i in range(iters):
        all_vec = next_inputs[i].reshape((1, -1))
        smiles = model.sample(1, z=torch.tensor(all_vec).float())[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        err = Chem.SanitizeMol(mol, catchErrors=True)
        if err != 0:
            continue
        valid_smiles.append(smiles)
        new_features.append(all_vec)

    valid_smiles = valid_smiles[:50]
    if len(new_features) != 0:
        new_features = np.vstack(new_features)[:50]
    else:
        new_features = np.zeros((0, X_train.shape[1]))
    os.makedirs(args.save_dir, exist_ok=True)
    save_object(valid_smiles,
                os.path.join(args.save_dir,
                             "valid_smiles{}.dat".format(iteration)))

    scores = []
    for i in range(len(valid_smiles)):
        mol = MolFromSmiles(valid_smiles[i])
        current_log_P_value = Descriptors.MolLogP(mol)
        current_SA_score = -SA(mol)
        current_cycle_score = max_ring_penalty(mol)

        current_SA_score_normalized = (current_SA_score - np.mean(
            SA_scores)) / np.std(SA_scores)
        current_log_P_value_normalized = (current_log_P_value - np.mean(
            logP_values)) / np.std(logP_values)
        current_cycle_score_normalized = (current_cycle_score - np.mean(
            cycle_scores)) / np.std(cycle_scores)

        score = (current_SA_score_normalized +
                 current_log_P_value_normalized +
                 current_cycle_score_normalized)
        scores.append(-score)  # target is always minused

    print(f"{len(valid_smiles)} molecules found. Scores: {scores}")
    save_object(scores,
                os.path.join(args.save_dir, "scores{}.dat".format(iteration)))

    if len(new_features) > 0:
        X_train = np.concatenate([X_train, new_features], 0)
        y_train = np.concatenate([y_train, np.array(scores)[:, None]], 0)

    iteration += 1
