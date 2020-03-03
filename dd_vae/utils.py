import random
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus


def prepare_seed(seed=777, n_jobs=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_jobs)


def smoothed_log_indicator(x, temperature):
    return softplus(-x/temperature + np.log(1/temperature - 1))


def combine_loss(loss_components, weights):
    if len(weights) == 0:
        raise ValueError("Specify at least one weight")
    loss = 0
    for component in weights:
        loss += loss_components[component] * weights[component]
    return loss


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        shape = shape or [-1]
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'

    def save(self, path):
        device = self.device
        self.to('cpu')
        weights = self.state_dict()
        data = {
            'weights': weights,
            'config': self.config
        }
        torch.save(data, path)
        self.to(device)

    @classmethod
    def load(cls, path, restore_weights=True):
        data = torch.load(path)
        model = cls(**data['config'])
        if restore_weights:
            model.load_state_dict(data['weights'])
        return model

    def to(self, device):
        self.device = device
        super().to(device)
        return self


class LinearGrowth:
    def __init__(self, start, end, start_epoch, end_epoch, log=False):
        self.start = start
        self.end = end
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.log = log
        if log:
            self.start = np.log10(start)
            self.end = np.log10(end)

    def __call__(self, epoch):
        if epoch <= self.start_epoch:
            value = self.start
        elif epoch >= self.end_epoch:
            value = self.end
        else:
            delta = (self.end - self.start) / (
                self.end_epoch - self.start_epoch)
            value = delta * (epoch - self.start_epoch) + self.start
        if self.log:
            value = 10**value
        return value


def to_onehot(x, n):
    one_hot = torch.zeros(x.shape[0], n)
    one_hot.scatter_(1, x[:, None].cpu(), 1)
    one_hot = one_hot.to(x.device)
    return one_hot


class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'


class CharVocab:
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)

        return cls(chars, *args, **kwargs)

    def __init__(self, chars, st=SpecialTokens):
        if (st.bos in chars) or (st.eos in chars) or \
                (st.pad in chars) or (st.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [st.bos, st.eos, st.pad, st.unk]

        self.st = st
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.st.bos]

    @property
    def eos(self):
        return self.c2i[self.st.eos]

    @property
    def pad(self):
        return self.c2i[self.st.pad]

    @property
    def unk(self):
        return self.c2i[self.st.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.st.unk

        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string


class StringDataset:
    def __init__(self, vocab, data):
        self.tokens = [vocab.string2ids(s) for s in data]
        self.data = data
        self.bos = vocab.bos
        self.eos = vocab.eos

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        with_bos = torch.tensor([self.bos] + tokens, dtype=torch.long)
        with_eos = torch.tensor(tokens + [self.eos], dtype=torch.long)
        return with_bos, with_eos, self.data[index]


def collate(batch, pad, return_data=False):
    with_bos, with_eos, data = list(zip(*batch))
    lengths = [len(x) for x in with_bos]
    order = np.argsort(lengths)[::-1]
    with_bos = [with_bos[i] for i in order]
    with_eos = [with_eos[i] for i in order]
    lengths = [lengths[i] for i in order]
    with_bos = torch.nn.utils.rnn.pad_sequence(
        with_bos, padding_value=pad
    )
    with_eos = torch.nn.utils.rnn.pad_sequence(
        with_eos, padding_value=pad
    )
    if return_data:
        data = np.array(data)[order]
        return with_bos, with_eos, lengths, data
    return with_bos, with_eos, lengths


def batch_to_device(batch, device):
    return [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in batch
    ]
