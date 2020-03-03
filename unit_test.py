from unittest import TestCase
import numpy as np
import torch

from dd_vae.proposals import get_proposals


class TestUtils(TestCase):
    def test_kl(self):
        np.random.seed(0)
        batch_size = 100000
        m = np.random.randn(32, 1)/3
        s = np.exp(np.random.randn(32, 1))/3
        m_rep = torch.tensor(np.tile(m, (1, batch_size)))
        s_rep = torch.tensor(np.tile(s, (1, batch_size)))
        m_t = torch.tensor(m)
        s_t = torch.tensor(s)
        for name, proposal_class in get_proposals().items():
            with self.subTest(name=name):
                proposal = proposal_class()
                samples = proposal.sample(m_rep, s_rep)
                density = proposal.density((samples-m_rep) / s_rep) / s_rep
                kl = proposal.kl(m_t, s_t)
                kl_mc = (
                    np.log(density * np.sqrt(2 * np.pi)) +
                    samples**2 / 2
                ).mean(1)
                self.assertLess(
                    torch.abs(kl - kl_mc).max().item(), 0.05,
                    f"Failed proposal {name} for Gaussian prior"
                )

    def test_kl_uniform(self):
        np.random.seed(0)
        batch_size = 100000
        m = np.random.randn(32, 1)/3
        s = np.exp(np.random.randn(32, 1))/3
        m_rep = torch.tensor(np.tile(m, (1, batch_size)))
        s_rep = torch.tensor(np.tile(s, (1, batch_size)))
        m_t = torch.tensor(m)
        s_t = torch.tensor(s)
        for name, proposal_class in get_proposals().items():
            if name == 'gaussian':
                continue
            with self.subTest(name=name):
                proposal = proposal_class()
                samples = proposal.sample(m_rep, s_rep)
                density = proposal.density((samples-m_rep) / s_rep) / s_rep
                kl = proposal.kl_uniform(m_t, s_t)
                kl_mc = np.log(density * 2).mean(1)
                self.assertLess(
                    torch.abs(kl - kl_mc).max().item(), 0.05,
                    f"Failed proposal {name} for uniform prior"
                )
