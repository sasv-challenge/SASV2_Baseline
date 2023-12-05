import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from typing import Tuple
from utils import PreEmphasis
from .specaugment import SpecAugment
from .wenet.transformer.encoder_cat import ConformerEncoder

class Conformer(nn.Module):
    def __init__(self, num_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", pos_enc_layer_type="rel_pos"):
        super(Conformer, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        self.conformer = ConformerEncoder(input_size=num_mels, num_blocks=num_blocks, output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type, )
        self.bn = nn.BatchNorm1d(output_size*num_blocks*2)
        self.fc = nn.Linear(output_size*num_blocks*2, embedding_dim)

        self.specaug = SpecAugment()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )
        output_dim = output_size*num_blocks
        self.attention = nn.Sequential(
            nn.Conv1d(output_dim*3, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, output_dim, kernel_size=1),
            nn.Softmax(dim=2),
            )

    def forward(self, x: Tensor, aug=False) -> Tuple[Tensor, bool]:

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x)+1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if aug == True:
                    x = self.specaug(x)
        x = x.transpose(1,2)
        lens = torch.ones(x.shape[0]).to(x.device)
        lens = torch.round(lens*x.shape[1]).int()
        x, masks = self.conformer(x, lens)
        x = x.transpose(1,2)

        # Context dependent ASP
        t = x.size()[-1]
        global_x = torch.cat((x,torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t), torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        x = torch.cat((mu, sg), dim=1)

        # BN -> FC: embedding
        x = self.bn(x)
        x = self.fc(x)

        return x

def MainModel(num_mels=80, num_out=192, **kwargs):
    model = Conformer(num_mels=num_mels, embedding_dim=num_out, input_layer="conv2d2")
    return model
