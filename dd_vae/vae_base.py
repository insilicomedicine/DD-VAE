import torch
from torch import nn
from .utils import BaseModel, batch_to_device
from .proposals import get_proposals


class VAE(BaseModel):
    """
    Generative Recurrent Autoregressive Decoder
    """
    def __init__(self, prior, proposal):
        super().__init__()
        if prior not in ['gaussian', 'uniform']:
            raise ValueError(
                "Supported priors are 'gaussian' and 'uniform'")
        if proposal not in get_proposals():
            proposals = list(get_proposals().keys())
            raise ValueError(
                f"Supported proposals are {proposals}")

        self.config = {
            'proposal': proposal,
            'prior': prior
        }
        self.proposal = get_proposals()[proposal]()
        self.prior = prior

    def encode(self, batch):
        """
        Encodes batch and returns latent codes
        """
        raise NotImplementedError

    def decode(self, batch, z=None, state=None):
        """
        Decodes batch and returns logits and intermediate states
        """
        raise NotImplementedError

    def compute_metrics(self, batch, logits):
        return {}

    def language_model_nll(self, with_eos, logits):
        loss = nn.NLLLoss(ignore_index=self.vocab.pad)(
            logits.transpose(1, 2), with_eos)
        return loss

    def encoder_parameters(self):
        raise NotImplementedError

    def decoder_parameters(self):
        raise NotImplementedError

    def get_mu_std(self, z):
        dim = z.shape[1] // 2
        mu, logstd = z.split(dim, 1)
        std = logstd.exp()

        if self.prior == 'uniform':
            left = torch.sigmoid(mu - std)
            right = torch.sigmoid(mu + std)
            mu = (right + left) - 1
            std = (right - left)

        return mu, std

    def sample_kl(self, z, mu_only=False):
        mu, std = self.get_mu_std(z)
        if self.prior == 'gaussian':
            kl_loss = self.proposal.kl(mu, std).mean()
        elif self.prior == 'uniform':
            kl_loss = self.proposal.kl_uniform(mu, std).mean()
        else:
            raise ValueError

        if mu_only:
            sample = mu
        else:
            sample = self.proposal.sample(mu, std)
        return sample, kl_loss

    def argmax_nll(self, batch, logits, temperature):
        raise NotImplementedError

    def sample_nll(self, batch, logits):
        raise NotImplementedError

    def get_loss_components(self, batch, temperature):
        batch = batch_to_device(batch, self.device)
        z = self.encode(batch)
        sample, kl_loss = self.sample_kl(z, not self.variational)
        logits, _ = self.decode(batch, z=sample)
        metrics = self.compute_metrics(batch, logits)
        language_model_nll = self.sample_nll(batch, logits)
        argmax_nll = self.argmax_nll(batch, logits, temperature)
        loss_components = {
            'sample_nll': language_model_nll,
            'kl_loss': kl_loss,
            'argmax_nll': argmax_nll,
            **metrics
        }
        return loss_components

    def sample(self, batch_size=1, mode='argmax', z=None):
        raise NotImplementedError
