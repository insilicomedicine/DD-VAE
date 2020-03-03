import argparse
import ast
import sys
from functools import partial
from configparser import ConfigParser
from tqdm import tqdm
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.datasets import MNIST

from dd_vae.vae_mnist import VAE_MNIST
from dd_vae.vae_rnn import VAE_RNN
from dd_vae.utils import CharVocab, collate, StringDataset, \
                  LinearGrowth, combine_loss, prepare_seed
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR, StepLR


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--device',
                        type=str, required=True, default='cpu',
                        help='Training device')
    parsed_args, errors = parser.parse_known_args(args[1:])
    if len(errors) != 0:
        raise ValueError(f"Unknown arguments {errors}")
    return parsed_args


def infer_config_types(parameters):
    return dict({
        k: ast.literal_eval(v)
        for k, v in parameters.items()
    })


def parse_config(config_path):
    config = ConfigParser()
    paths = config.read(config_path)
    if len(paths) == 0:
        raise ValueError(f"Config file {config_path} does not exist")

    infered_config = {
        k: infer_config_types(v)
        for k, v in config.items()
    }
    return infered_config


def add_dict(left, right):
    for key, value in right.items():
        left[key] = left.get(key, 0) + value.item()


def train_epoch(model, loss_weights, epoch, data_loader,
                backward, temperature, logger,
                optimizer, verbose=True, clamp=None, fine_tune=False):
    if backward:
        label = '/train'
    else:
        label = '/test'
    total_loss = {}
    iterations = 0
    for batch in tqdm(
            data_loader, postfix=f'Epoch {epoch} {label}',
            disable=not verbose):
        iterations += 1
        loss_components = model.get_loss_components(batch, temperature)
        loss = combine_loss(loss_components, loss_weights)
        loss_components['loss'] = loss
        add_dict(total_loss, loss_components)
        if backward:
            optimizer['encoder'].zero_grad()
            optimizer['decoder'].zero_grad()
            loss.backward()
            if clamp is not None:
                for param in model.parameters():
                    param.grad.clamp_(-clamp, clamp)
            if not fine_tune:
                optimizer['encoder'].step()
            optimizer['decoder'].step()

    for key, value in total_loss.items():
        logger.add_scalar(key + label,
                          value / iterations,
                          global_step=epoch)


def prepare_mnist(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.3).float())
    ])
    train_dataset = MNIST('data/mnist/', train=True,
                          download=True, transform=transform)
    test_dataset = MNIST('data/mnist/', train=False,
                         transform=transform)
    batch_size = config['train']['batch_size']
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    if 'load' in config['model']:
        model = VAE_MNIST.load(config['model']['load'])
    else:
        model = VAE_MNIST(**config['model'])
    return train_loader, test_loader, model


def load_csv(path):
    if path.endswith('.csv'):
        return [x.strip() for x in open(path)]
    if path.endswith('.csv.gz'):
        df = pd.read_csv(path, compression='gzip',
                         dtype='str', header=None)
        return list(df[0].values)
    raise ValueError("Unknown format")


def prepare_rnn(config):
    data_config = config['data']
    train_config = config['train']
    train_data = load_csv(data_config['train_path'])
    vocab = CharVocab.from_data(train_data)
    if 'load' in config['model']:
        print("LOADING")
        model = VAE_RNN.load(config['model']['load'])
        vocab = model.vocab
    else:
        model = VAE_RNN(vocab=vocab, **config['model'])
    collate_pad = partial(collate, pad=vocab.pad)
    train_dataset = StringDataset(vocab, train_data)
    train_loader = DataLoader(
        train_dataset, collate_fn=collate_pad,
        batch_size=train_config['batch_size'], shuffle=True)
    if 'test_path' in data_config:
        test_data = load_csv(data_config['test_path'])
        test_dataset = StringDataset(vocab, test_data)
        test_loader = DataLoader(
            test_dataset, collate_fn=collate_pad,
            batch_size=train_config['batch_size'])
    else:
        test_loader = None
    return train_loader, test_loader, model


def train(config_path, device):
    """
    Trains a deterministic VAE model.

    Parameters:
        config_path: path to .ini file with model configuration
        device: device for training ('cpu' for CPU, 'cuda:n' for GPU #n)
        train_data: list of train dataset strings
        test_data: list of test dataset strings
    """
    config = parse_config(config_path)
    prepare_seed(seed=config['train'].get('seed', 777))

    data_config = config['data']
    if data_config['title'].lower() == 'mnist':
        train_loader, test_loader, model = prepare_mnist(config)
    else:
        train_loader, test_loader, model = prepare_rnn(config)

    model = model.to(device)

    train_config = config['train']
    save_config = config['save']
    kl_config = config['kl']
    temperature_config = config['temperature']
    model_dir = save_config['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(save_config['log_dir'], exist_ok=True)

    optimizer = {
        'encoder': torch.optim.Adam(model.encoder_parameters(),
                                    lr=train_config['lr']),
        'decoder': torch.optim.Adam(model.decoder_parameters(),
                                    lr=train_config['lr'])
    }
    scheduler_class = (
        MultiStepLR
        if isinstance(train_config['lr_reduce_epochs'], (list, tuple))
        else StepLR
    )
    scheduler = {
        'encoder': scheduler_class(
                optimizer['encoder'],
                train_config['lr_reduce_epochs'],
                train_config['lr_reduce_gamma']),
        'decoder': scheduler_class(
                optimizer['decoder'],
                train_config['lr_reduce_epochs'],
                train_config['lr_reduce_gamma'])
    }

    logger = SummaryWriter(save_config['log_dir'])

    kl_weight = LinearGrowth(**kl_config)
    temperature = LinearGrowth(**temperature_config)
    epoch_verbose = train_config.get('verbose', None) == 'epoch'
    batch_verbose = not epoch_verbose

    pretrain = train_config.get('pretrain', 0)
    if pretrain != 0:
        pretrain_weight = LinearGrowth(0, 1, 0, pretrain)
    fine_tune = train_config.get('fune_tune', 0)
    for epoch in tqdm(range(train_config['epochs'] + pretrain + fine_tune),
                      disable=not epoch_verbose):
        fine_tune = epoch >= train_config['epochs'] + pretrain
        current_temperature = temperature(epoch)
        if epoch < pretrain:
            w = pretrain_weight(epoch)
            loss_weights = {'argmax_nll': w,
                            'sample_nll': 1 - w}
        elif train_config['mode'] == 'argmax':
            loss_weights = {'argmax_nll': 1}
            logger.add_scalar('temperature', current_temperature, epoch)
        else:
            loss_weights = {'sample_nll': 1}
        loss_weights['kl_loss'] = kl_weight(epoch)
        logger.add_scalar('kl_weight', loss_weights['kl_loss'], epoch)

        scheduler['encoder'].step()
        scheduler['decoder'].step()

        train_epoch(
            model, loss_weights, epoch, train_loader, True,
            current_temperature, logger,
            optimizer, batch_verbose,
            clamp=train_config.get('clamp'),
            fine_tune=fine_tune
        )

        if test_loader is not None:
            with torch.no_grad():
                train_epoch(
                    model, loss_weights, epoch, test_loader, False,
                    current_temperature, logger,
                    optimizer, batch_verbose,
                    clamp=train_config.get('clamp'),
                    fine_tune=fine_tune
                )
        if train_config.get("checkpoint", "epoch") == "epoch":
            path = f"{model_dir}/checkpoint_{epoch+1}.pt"
        else:
            path = f"{model_dir}/checkpoint.pt"
        model.save(path)

    model.save(f"{model_dir}/checkpoint.pt")
    logger.close()


if __name__ == "__main__":
    parsed_args = parse_args(sys.argv)
    train(parsed_args.config, parsed_args.device)
