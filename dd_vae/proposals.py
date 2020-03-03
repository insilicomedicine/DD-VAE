import numpy as np
import torch


def get_proposals():
    '''
    Returns a dict with implemented finite support proposals
    '''
    return {
        'uniform': UniformProposal,
        'triangular': TriangularProposal,
        'epanechnikov': EpanechnikovProposal,
        'quartic': QuarticProposal,
        'triweight': TriweightProposal,
        'tricube': TricubeProposal,
        'cosine': CosineProposal,
        'gaussian': GaussianProposal
    }


class Proposal:
    def __init__(self, eps=1e-9):
        self.buffer = np.zeros(0)
        self.eps = eps

    def density(self, z):
        raise NotImplementedError

    def kl(self, m=0, s=1):
        raise NotImplementedError

    def kl_uniform(self, m=0, s=1):
        raise NotImplementedError

    def sample(self, m, s):
        batch_size = np.prod(m.shape)
        uniform_height = self.density(0)
        acceptance_rate = 0.5 / uniform_height
        up_batch_size = int(batch_size / acceptance_rate) + 1
        while self.buffer.shape[0] < batch_size:
            sample, rejection_sample = np.split(
                np.random.rand(2*up_batch_size), 2
            )
            sample = sample * 2 - 1
            rejection_sample = rejection_sample * uniform_height
            density = self.density(sample)
            sample = sample[rejection_sample < density]
            self.buffer = np.concatenate((self.buffer, sample), 0)

        sample = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]
        sample = sample.reshape(*m.shape)
        sample = torch.tensor(sample, dtype=m.dtype, device=m.device)
        return sample * s + m


class UniformProposal(Proposal):
    def density(self, z):
        return z * 0 + 0.5

    def sample(self, m, s):
        sample = np.random.rand(*m.shape) * 2 - 1
        sample = torch.tensor(sample, dtype=m.dtype, device=m.device)
        return sample * s + m

    def kl(self, m, s):
        return (0.5 * m**2 + s**2/6 - torch.log(s+self.eps) +
                0.5*np.log(2*np.pi) - np.log(2)).sum(1)

    def kl_uniform(self, m, s):
        return (-torch.log(s+self.eps)).sum(1)


class TriangularProposal(Proposal):
    def density(self, z):
        return 1 - np.abs(z)

    def kl(self, m, s):
        return (0.5 * m**2 + s**2/12 - torch.log(s+self.eps) +
                0.5*np.log(2*np.pi) - 0.5).sum(1)

    def kl_uniform(self, m, s):
        return (-0.5 + np.log(2) - torch.log(s+self.eps)).sum(1)


class EpanechnikovProposal(Proposal):
    def density(self, z):
        return 0.75 * (1 - z**2)

    def kl(self, m, s):
        return (0.5 * m**2 + s**2/10 - torch.log(s+self.eps) +
                0.5*np.log(2*np.pi) - 5/3 + np.log(3)).sum(1)

    def kl_uniform(self, m, s):
        return (-5/3 + np.log(6)-torch.log(s+self.eps)).sum(1)


class QuarticProposal(Proposal):
    def density(self, z):
        return 15/16 * (1 - z**2)**2

    def kl(self, m, s):
        return (0.5 * m**2 + s**2/14 - torch.log(s+self.eps) +
                0.5*np.log(2*np.pi) - 47/15 + np.log(15)).sum(1)

    def kl_uniform(self, m, s):
        return (-47/15 + np.log(30) - torch.log(s+self.eps)).sum(1)


class TriweightProposal(Proposal):
    def density(self, z):
        return 35/32 * (1 - z**2)**3

    def kl(self, m, s):
        return (0.5 * m**2 + s**2/18 - torch.log(s+self.eps) +
                0.5*np.log(2*np.pi) - 319/70 + np.log(70)).sum(1)

    def kl_uniform(self, m, s):
        return (-319/70 + np.log(140) - torch.log(s+self.eps)).sum(1)


class TricubeProposal(Proposal):
    def density(self, z):
        return 70/81 * (1 - np.abs(z)**3)**3

    def kl(self, m, s):
        return (0.5 * m**2 + 35*s**2/486 - torch.log(s+self.eps) +
                0.5*np.log(2*np.pi) + np.pi * np.sqrt(3) / 2 -
                1111/140 + np.log(70*np.sqrt(3))).sum(1)

    def kl_uniform(self, m, s):
        return (np.pi * np.sqrt(3) / 2 - 1111/140 +
                np.log(140*np.sqrt(3)) - torch.log(s+self.eps)).sum(1)


class CosineProposal(Proposal):
    def density(self, z):
        return np.pi/4 * np.cos(np.pi * z / 2)

    def kl(self, m, s):
        return (0.5 * m**2 + (0.5 - 4 / np.pi**2)*s**2 -
                torch.log(s+self.eps) + 0.5*np.log(2*np.pi) -
                1 + np.log(np.pi/2)).sum(1)

    def kl_uniform(self, m, s):
        return (-1 + np.log(np.pi) - torch.log(s+self.eps)).sum(1)


class GaussianProposal(Proposal):
    def density(self, z):
        return np.exp(-(z**2)/2) / np.sqrt(2 * np.pi)

    def sample(self, m, s):
        sample = torch.randn(*m.shape, dtype=m.dtype, device=m.device)
        return sample * s + m

    def kl(self, m, s):
        return 0.5 * (m**2 + s**2 - 2 * torch.log(s+self.eps) - 1).sum(1)

    def kl_uniform(self, m, s):
        raise ValueError("KL(N || U) = -inf")
