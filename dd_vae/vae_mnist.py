import torch
from torch import nn
from .utils import smoothed_log_indicator, Reshape
from .vae_base import VAE


class VAE_MNIST(VAE):
    def __init__(self, layer_sizes, latent_size,
                 proposal='tricube',
                 prior='gaussian',
                 variational=True,
                 image_size=28,
                 channels=1):
        super().__init__(prior=prior, proposal=proposal)
        self.config.update({
            'layer_sizes': layer_sizes,
            'latent_size': latent_size,
            'variational': variational,
            'image_size': image_size,
            'channels': channels
        })

        self.encoder = nn.Sequential(
            Reshape(784),
            *self.DNN(784, *layer_sizes, 2*latent_size)
        )

        self.decoder = nn.Sequential(
            *self.DNN(latent_size, *layer_sizes[::-1], 784),
            Reshape(1, 28, 28),
            nn.Sigmoid()
        )
        self.latent_size = latent_size
        self.variational = variational

    @staticmethod
    def DNN(*layers):
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 2:
                net.append(nn.LeakyReLU())
        return net

    def compute_metrics(self, batch, logits):
        images, _ = batch
        match = (images.long() == (logits > 0.5).long())
        match = match.view(match.shape[0], -1).float()
        return {
            'pixel_accuracy': match.mean(),
            'image_accuracy': match.min(1)[0].mean(),
            'image_accuracy@10': ((1-match).sum(1) < 10).float().mean()
        }

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()

    def sample_nll(self, batch, logits):
        images, _ = batch
        return torch.nn.BCELoss()(logits, images)

    def encode(self, batch):
        image, _ = batch
        return self.encoder(image.float())

    def decode(self, batch, z=None, state=None):
        return self.decoder(z), None

    def argmax_nll(self, batch, logits, temperature):
        images, _ = batch
        p_correct = logits*images + (1 - logits)*(1 - images)
        delta = p_correct - (1 - p_correct)
        reconstruction_loss = smoothed_log_indicator(delta, temperature).mean()
        return reconstruction_loss

    def sample(self, batch_size=1, z=None):
        if z is None:
            if self.prior == 'gaussian':
                z = torch.randn(batch_size, self.latent_size)
            elif self.prior == 'uniform':
                z = torch.rand(batch_size, self.latent_size)*2 - 1
        z = z.to(self.device)
        return self.decoder(z)
