import torch
from torch import nn
import numpy as np
from .utils import to_onehot, smoothed_log_indicator
from .proposals import get_proposals
from .vae_base import VAE


class VAE_RNN(VAE):
    def __init__(self, embedding_size,
                 hidden_size, latent_size,
                 num_layers, vocab,
                 proposal='tricube',
                 prior='gaussian',
                 variational=True,
                 use_embedding_input=True,
                 fc=None,
                 fc_norm=False):
        super().__init__(prior=prior, proposal=proposal)
        self.vocab = vocab
        self.config.update({
            'embedding_size': embedding_size,
            'hidden_size': hidden_size,
            'latent_size': latent_size,
            'num_layers': num_layers,
            'proposal': proposal,
            'prior': prior,
            'vocab': self.vocab,
            'variational': variational,
            'use_embedding_input': use_embedding_input,
            'fc': fc,
            'fc_norm': fc_norm
        })

        self.vocab_size = len(self.vocab)
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.encoder_embedding = nn.Embedding(self.vocab_size,
                                              embedding_size,
                                              self.vocab.pad)
        rnn_input_size = embedding_size
        self.encoder = nn.GRU(rnn_input_size, hidden_size,
                              num_layers=num_layers)
        self.encoder_to_latent = self.get_fc(
            fc, hidden_size * num_layers, 2 * latent_size, fc_norm)
        self.latent_to_decoder = self.get_fc(
            fc, latent_size, hidden_size * num_layers, fc_norm)
        if use_embedding_input:
            self.decoder_embedding = nn.Embedding(self.vocab_size,
                                                  embedding_size,
                                                  self.vocab.pad)
            decoder_input_size = rnn_input_size
        else:
            self.decoder_embedding = None
            decoder_input_size = 1

        self.decoder = nn.GRU(decoder_input_size, hidden_size,
                              num_layers=num_layers)

        self.proposal = get_proposals()[proposal]()
        self.prior = prior

        self.decoder_to_logits = nn.Linear(
            hidden_size, self.vocab_size
        )

        self.variational = variational
        self.use_embedding_input = use_embedding_input

    @staticmethod
    def get_fc(layers, input_dim, output_dim, fc_norm):
        if layers is None:
            return nn.Linear(input_dim, output_dim)
        layers = [input_dim] + layers + [output_dim]
        network = []
        for i in range(len(layers) - 2):
            network.append(nn.Linear(layers[i], layers[i+1]))
            if fc_norm:
                network.append(nn.LayerNorm(layers[i+1]))
            network.append(nn.ELU())
        network.append(nn.Linear(layers[-2], layers[-1]))
        return nn.Sequential(*network)

    def encode(self, batch):
        with_bos, with_eos, lengths = batch
        emb = self.encoder_embedding(with_eos)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(emb, lengths)
        _, h = self.encoder(packed_sequence, None)
        h = h.transpose(0, 1).contiguous().view(h.shape[1], -1)
        z = self.encoder_to_latent(h)
        return z

    def decode(self, batch, z=None, state=None):
        with_bos, with_eos, lengths = batch
        if state is None:
            state = self.latent_to_decoder(z)
            state = state.view(
                state.shape[0], self.num_layers, -1
            ).transpose(0, 1).contiguous()

        if self.use_embedding_input:
            emb = self.decoder_embedding(with_bos)
        else:
            emb = torch.zeros(
                (with_bos.shape[0], with_bos.shape[1], 1),
                device=with_bos.device
            )

        packed_sequence = nn.utils.rnn.pack_padded_sequence(emb, lengths)
        states, state = self.decoder(packed_sequence, state)
        states, _ = nn.utils.rnn.pad_packed_sequence(states)
        logits = self.decoder_to_logits(states)
        logits = torch.log_softmax(logits, 2)
        return logits, state

    def compute_metrics(self, batch, logits):
        with_bos, with_eos, lengths = batch
        predictions = torch.argmax(logits, 2)
        pad_mask = (with_eos == self.vocab.pad)
        non_pad_mask = (~pad_mask).float()
        correct_prediction = (predictions == with_eos)
        string_accuracy = (
            correct_prediction | pad_mask
        ).float().min(0)[0].mean()
        character_accuracy = (
            correct_prediction.float() * non_pad_mask
        ).sum() / non_pad_mask.sum()
        return {
            'string_accuracy': string_accuracy,
            'character_accuracy': character_accuracy
        }

    def sample_nll(self, batch, logits):
        with_bos, with_eos, lengths = batch
        loss = nn.NLLLoss(
            ignore_index=self.vocab.pad,
            reduction='mean')(logits.transpose(1, 2), with_eos)
        return loss

    def encoder_parameters(self):
        return nn.ModuleList([self.encoder_embedding,
                              self.encoder,
                              self.encoder_to_latent]).parameters()

    def decoder_parameters(self):
        modules = [self.decoder,
                   self.latent_to_decoder,
                   self.decoder_to_logits]
        if self.use_embedding_input:
            modules.append(self.decoder_embedding)
        return nn.ModuleList(modules).parameters()

    def argmax_nll(self, batch, logits, temperature):
        with_bos, with_eos, lengths = batch
        with_eos = with_eos.view(-1)

        logits = logits.view(-1, logits.shape[2])
        oh = to_onehot(with_eos, logits.shape[1])
        delta = (logits * oh).sum(1, keepdim=True) - logits
        error = smoothed_log_indicator(delta, temperature) * (1 - oh)
        error = error.sum(1)
        pad_mask = (with_eos != self.vocab.pad).float()
        error = error * pad_mask
        error = error.mean() / pad_mask.mean()
        return error

    def sample(self, batch_size=1, max_len=100, mode='argmax',
               z=None, keep_stats=False,
               temperature=1):
        if mode not in ['sample', 'argmax']:
            raise ValueError("Can either sample or argmax")
        generated_sequence = []
        if z is None:
            if self.prior == 'gaussian':
                z = torch.randn(batch_size, self.latent_size)
            elif self.prior == 'uniform':
                z = torch.rand(batch_size, self.latent_size)*2 - 1
        batch_size = z.shape[0]
        character = [[self.vocab.bos for _ in range(batch_size)]]
        character = torch.tensor(character, dtype=torch.long,
                                 device=self.device)
        h = self.latent_to_decoder(z.to(self.device))
        h = h.view(
                h.shape[0], self.num_layers, -1
        ).transpose(0, 1).contiguous()
        if keep_stats:
            stats = []
        lengths = [1]*batch_size
        for i in range(max_len):
            batch = (character, None, lengths)
            logits, h = self.decode(batch, state=h)
            if keep_stats:
                stats.append([logits.detach().cpu().numpy()])
            if mode == 'argmax':
                character = torch.argmax(logits[0], 1)
            else:
                character = torch.distributions.Categorical(
                    torch.exp(logits[0])).sample()
            character = character.detach()[None, :]
            generated_sequence.append(character.cpu().numpy())
        generated_sequence = np.concatenate(generated_sequence, 0).T
        samples = [self.vocab.ids2string(s) for s in generated_sequence]
        eos = self.vocab.i2c[self.vocab.eos]
        samples = [x.split(eos)[0] for x in samples]
        if keep_stats:
            return samples, stats
        else:
            return samples
