import argparse
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import rdkit
from dd_vae.bo.utils import max_ring_penalty
from dd_vae.utils import collate, StringDataset, batch_to_device
from dd_vae.vae_rnn import VAE_RNN
from moses.metrics import SA
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
rdkit.rdBase.DisableLog('rdApp.*')


def load_csv(path):
    if path.endswith('.gz'):
        df = pd.read_csv(path, compression='gzip',
                         dtype='str', header=None)
        return list(df[0].values)
    return [x.strip() for x in open(path)]


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--save_dir", type=str, required=True)
args = parser.parse_args(sys.argv[1:])

model = VAE_RNN.load(args.model)
model = model.to(args.device)

smiles = load_csv(args.data)

logP_values = []
latent_points = []
cycle_scores = []
SA_scores = []

print("Preparing dataset...")
collate_pad = partial(collate, pad=model.vocab.pad, return_data=True)
dataset = StringDataset(model.vocab, smiles)
data_loader = DataLoader(dataset, collate_fn=collate_pad,
                         batch_size=512, shuffle=False)
print("Getting latent codes...")
for batch in tqdm(data_loader):
    z = model.encode(batch_to_device(batch[:-1], args.device))
    mu, _ = model.get_mu_std(z)
    latent_points.append(mu.detach().cpu().numpy())
    romol = [Chem.MolFromSmiles(x.strip()) for x in batch[-1]]
    logP_values.extend([Descriptors.MolLogP(m) for m in romol])
    SA_scores.extend([-SA(m) for m in romol])
    cycle_scores.extend([max_ring_penalty(m) for m in romol])

SA_scores = np.array(SA_scores)
logP_values = np.array(logP_values)
cycle_scores = np.array(cycle_scores)

SA_scores_normalized = (SA_scores - SA_scores.mean()) / SA_scores.std()
logP_values_normalized = (logP_values - logP_values.mean()) / logP_values.std()
cycle_scores_normalized = (
    cycle_scores - cycle_scores.mean()) / cycle_scores.std()

latent_points = np.vstack(latent_points)

targets = (SA_scores_normalized +
           logP_values_normalized +
           cycle_scores_normalized)
os.makedirs(args.save_dir, exist_ok=True)
np.savez_compressed(os.path.join(args.save_dir, 'features.npz'),
                    latent_points=latent_points,
                    targets=targets, logP_values=logP_values,
                    SA_scores=SA_scores, cycle_scores=cycle_scores)
