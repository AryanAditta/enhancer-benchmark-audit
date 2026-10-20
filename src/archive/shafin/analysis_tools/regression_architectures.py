"""Architecture-only extraction; no training or notebook side effects. Source/license index accompanies this file."""
import math
import torch
from torch import nn
import torch.nn.functional as F

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class SELayer(nn.Module):

    def __init__(self, inp, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(inp, int(inp // reduction)), nn.SiLU(), nn.Linear(int(inp // reduction), inp), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class EffBlock(nn.Module):

    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim = self.in_ch * self.resize_factor
        self.block = nn.Sequential(nn.Conv1d(self.in_ch, self.inner_dim, kernel_size=1, padding='same', bias=False), nn.BatchNorm1d(self.inner_dim), activation(), nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=ks, groups=self.inner_dim, padding='same', bias=False), nn.BatchNorm1d(self.inner_dim), activation(), SELayer(self.inner_dim, reduction=self.se_reduction), nn.Conv1d(self.inner_dim, self.in_ch, kernel_size=1, padding='same', bias=False), nn.BatchNorm1d(self.in_ch), activation())

    def forward(self, x):
        return self.block(x)

class LocalBlock(nn.Module):

    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks
        self.block = nn.Sequential(nn.Conv1d(self.in_ch, self.out_ch, kernel_size=self.ks, padding='same', bias=False), nn.BatchNorm1d(self.out_ch), activation())

    def forward(self, x):
        return self.block(x)

class ResidualConcat(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)

class MapperBlock(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(nn.BatchNorm1d(in_features), nn.Conv1d(in_features, out_features, kernel_size=1))

    def forward(self, x):
        return self.block(x)

class LegNet(nn.Module):

    def __init__(self, in_ch, stem_ch, stem_ks, ef_ks, ef_block_sizes, pool_sizes, resize_factor, activation=nn.SiLU):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)
        self.in_ch = in_ch
        self.stem = LocalBlock(in_ch=in_ch, out_ch=stem_ch, ks=stem_ks, activation=activation)
        blocks = []
        in_ch = stem_ch
        out_ch = stem_ch
        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blocks.append(nn.Sequential(ResidualConcat(EffBlock(in_ch=in_ch, out_ch=in_ch, ks=ef_ks, resize_factor=resize_factor, activation=activation)), LocalBlock(in_ch=in_ch * 2, out_ch=out_ch, ks=ef_ks, activation=activation), nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()))
            in_ch = out_ch
        self.main = nn.Sequential(*blocks)
        self.mapper = MapperBlock(in_features=out_ch, out_features=out_ch * 2)
        self.head = nn.Sequential(nn.Linear(out_ch * 2, out_ch * 2), nn.BatchNorm1d(out_ch * 2), activation(), nn.Linear(out_ch * 2, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.head(x).squeeze(-1)

class MPRAnn(nn.Module):
    """PyTorch port of `standard()` in visze/sequence_cnn_models lib/model.py (MIT, Max Schubach / BIH).
    Keras Conv1D uses 'valid' padding; activation precedes BatchNormalization; softmax is over channels;
    Keras BN defaults (momentum 0.99, eps 1e-3) and glorot_uniform/zero initialisation are reproduced."""

    def __init__(self, length=200):
        super().__init__()
        bn = lambda c: nn.BatchNorm1d(c, eps=0.001, momentum=0.01)
        self.c1, self.b1 = (nn.Conv1d(4, 250, 7), bn(250))
        self.c2, self.b2 = (nn.Conv1d(250, 250, 8), bn(250))
        self.c3, self.b3 = (nn.Conv1d(250, 250, 3), bn(250))
        self.c4, self.b4 = (nn.Conv1d(250, 100, 2), bn(100))
        self.drop1, self.drop2, self.drop3 = (nn.Dropout(0.1), nn.Dropout(0.1), nn.Dropout(0.3))
        L = (length - 6 - 7) // 2 - 2 - 1
        self.d1, self.d2, self.out = (nn.Linear(100 * L, 300), nn.Linear(300, 200), nn.Linear(200, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.b1(F.relu(self.c1(x)))
        x = self.b2(torch.softmax(self.c2(x), dim=1))
        x = self.drop1(F.max_pool1d(x, 2))
        x = self.b3(torch.softmax(self.c3(x), dim=1))
        x = self.b4(torch.softmax(self.c4(x), dim=1))
        x = self.drop2(x)
        x = x.transpose(1, 2).flatten(1)
        x = self.drop3(torch.sigmoid(self.d1(x)))
        x = torch.sigmoid(self.d2(x))
        return self.out(x).squeeze(-1)

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, 64, 15, padding=7), nn.GELU(), nn.Conv1d(64, 96, 7, padding=3), nn.GELU(), nn.Conv1d(96, 96, 7, padding=3), nn.GELU())
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.conv(x)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(self.dropout(pooled)).squeeze(-1)

class Regressor(nn.Module):

    def __init__(self, enc):
        super().__init__()
        self.encoder = enc
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(int(enc.config.hidden_size), 1)

    def forward(self, b):
        output = self.encoder(**b)
        h = getattr(output, 'last_hidden_state', None)
        if h is None:
            h = output[0]
        mask = b['attention_mask'].to(h.dtype).unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.dropout(pooled)).squeeze(-1)
