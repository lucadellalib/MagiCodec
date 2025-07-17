import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from einops import rearrange

from magicodec.roformer import Roformer
from magicodec.roformer import ModelArgs as RoformerArgs


@dataclass
class ModelArgs:
    dim: int = 1024
    out_dim: int = 1024
    n_layers: int = 8
    n_heads: int = 16
    causal: bool = False
    use_qk_norm: str = 'head'
    use_window_mask: bool = True
    window_size: list = field(default_factory=lambda: [32, 0])
    window_type: str = "elemwise"  # elemwise, blockwise
    use_unet_style_skip_connect: bool = False
    flashattn_version: str = "2.8"


class VectorQuantize(nn.Module):
    def __init__(self, codebook_size, codebook_dim, loss_type="l2"):
        super().__init__()

        self.loss_type = loss_type

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.codebook_proj = nn.Linear(codebook_dim, codebook_dim)

    def forward(self, z_e):

        z_q, indices = self.decode_latents(z_e=z_e)  # [b, t, d]

        loss_codebook = F.l1_loss(z_q, z_e.detach())
        loss_commitment = F.l1_loss(z_e, z_q.detach())

        loss_vq = loss_codebook + loss_commitment * 0.25
        z_q = (z_e + (z_q - z_e).detach())

        return z_q, loss_vq, indices

    def inference(self, z_e):
        z_q, indices = self.decode_latents(z_e=z_e)  # [b, t, d]

        return z_q, indices

    def decode_latents(self, z_e):
        encodings = rearrange(z_e, "b t d -> (b t) d")
        codebook = self.codebook_proj(
            self.codebook.weight)  # codebook: (N x D)

        if self.loss_type == "l2":
            dist = torch.sum(encodings**2, dim = 1, keepdim = True) + \
                torch.sum(codebook**2, dim = 1) - 2 * \
                torch.einsum('bd,dn->bn', encodings, rearrange(codebook, 'n d -> d n'))
            indices = torch.argmin(dist, dim = 1)

        elif self.loss_type == "cos":
            norm_embed = F.normalize(encodings, dim = -1)
            norm_codebook = F.normalize(codebook, dim = -1)
            sim = torch.einsum(
                'bd,dn->bn', 
                norm_embed,
                rearrange(norm_codebook, 'n d -> d n')
            )
            indices = torch.argmax(sim, dim=1)

        indices = rearrange(indices, "(b t) -> b t", b=z_e.size(0))
        z_q = F.embedding(indices, codebook)

        return z_q, indices


class Encoder(nn.Module):
    def __init__(
        self,
        in_channel = 375,
        enc_channel = [768, 4096],
        out_channel = 16,
        hp = None,
    ):
        super().__init__()

        self.in_channel = in_channel
        self.linear_1 = torch.nn.Linear(in_channel, enc_channel[0], bias=False)
        self.linear_2 = torch.nn.Linear(enc_channel[0], enc_channel[1])

        self.transformer = Roformer(params=RoformerArgs(**vars(hp)))
        self.linear_out = torch.nn.Linear(enc_channel[1], out_channel)

    def forward(self, x):
        x = torch.reshape(x, [x.shape[0], -1, self.in_channel])
        x = self.linear_1(x)
        x = self.linear_2(x)
        with torch.autocast(
            device_type = "cuda",
            dtype = torch.bfloat16,
            enabled = True,
        ):
            x = self.transformer(x)

        x = self.linear_out(x.float())
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channel = 4096,
        dec_channel = [4096, 1024],
        out_channel = 375,
        hp = None,
    ):
        super().__init__()

        self.linear_in = torch.nn.Linear(in_channel, dec_channel[0])
        self.transformer = Roformer(params=RoformerArgs(**vars(hp)))
        self.linear_1 = torch.nn.Linear(dec_channel[0], dec_channel[1])
        self.linear_2 = torch.nn.Linear(
            dec_channel[1],
            out_channel,
            bias = False,
        )

    def forward(self, x):
        x = self.linear_in(x)

        with torch.autocast(
            device_type = "cuda",
            dtype = torch.bfloat16,
            enabled = True,
        ):
            x = self.transformer(x)

        x = self.linear_1(x.float())
        x = self.linear_2(x)
        x = torch.reshape(x, [x.shape[0], 1, -1])
        return x


class Generator(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        token_hz: int = 50,
    ):
        super().__init__()

        self.token_hz = token_hz

        if self.token_hz == 25:
            self.frame_len = 640
            self.enc_window_size = [16, 0]
            self.dec_window_size = [16, 1]  # [640ms, 40ms]

        if self.token_hz == 50:
            self.frame_len = 320
            self.enc_window_size = [32, 0]
            self.dec_window_size = [32, 2]  # [640ms, 40ms]

        if self.token_hz == 100:
            self.frame_len = 160
            self.enc_window_size = [64, 0]
            self.dec_window_size = [64, 4]  # [640ms, 40ms]

        enc_config = ModelArgs(
            dim = 1024,
            out_dim = 1024,
            n_layers = 8,
            n_heads = 16,
            window_size = self.enc_window_size,
        )

        dec_config = ModelArgs(
            dim = 1024,
            out_dim = 1024,
            n_layers = 8,
            n_heads = 16,
            window_size = self.dec_window_size
        )

        self.e1_dim = 768
        self.e2_dim = 1024
        self.d1_dim = 1024
        self.d2_dim = 768

        self.sample_rate = sample_rate
        self.codebook_size = 131072
        self.codebook_dim = 16

        self.encoder = Encoder(
            in_channel = self.frame_len,
            enc_channel = [self.e1_dim, self.e2_dim],
            out_channel = self.codebook_dim,
            hp = enc_config,
        )

        self.quantizer = VectorQuantize(
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
        )

        self.decoder = Decoder(
            in_channel = self.codebook_dim,
            dec_channel = [self.d1_dim, self.d2_dim],
            out_channel = self.frame_len,
            hp = dec_config,
        )

    def pad_audio(self, x):
        frame_len = self.frame_len
        length = x.shape[-1]
        frame_num = math.ceil(length / frame_len)
        pad_len = frame_num * frame_len - length
        x = F.pad(x, (0, pad_len), "constant", 0.0)
        return x

    def forward(self, x):
        with torch.autocast(
            device_type = "cuda",
            dtype = torch.bfloat16,
            enabled = True,
        ):
            x = self.pad_audio(x)
            z_e = self.encoder(x)  # [b, d, t]
            z_q, loss_vq, quantized_indices = self.quantizer(z_e)  # [b, t, d]
            loss_ze = torch.mean(torch.pow(z_e, 2))
            x_hat = self.decoder(z_q)
        
        return x_hat.float(), loss_vq, loss_ze, quantized_indices

    def sig_to_feats(self, x):
        x = self.pad_audio(x)
        z_e = self.encoder(x)
        return z_e

    def sig_to_qfeats(self, x):
        z_e = self.sig_to_feats(x)
        z_q = self.quantizer(z_e)[0]
        return z_q

    def sig_to_toks(self, x):
        z_e = self.sig_to_feats(x)
        quantized_indices = self.quantizer(z_e)[-1]
        return quantized_indices

    def feats_to_sig(self, z_e):
        x_hat = self.decoder(z_e)
        return x_hat

    def toks_to_sig(self, quantized_indices):
        codebook = self.quantizer.codebook_proj(self.quantizer.codebook.weight)
        z_q = F.embedding(quantized_indices, codebook)
        x_hat = self.feats_to_sig(z_q)[:, 0]
        return x_hat

    @torch.no_grad()
    def infer(self, x):
        with torch.autocast(
            device_type = "cuda",
            dtype = torch.bfloat16,
            enabled = True,
        ):
            x = self.pad_audio(x)
            z_e = self.encoder(x)
            z_q, quantized_indices = self.quantizer.inference(z_e)
            x_hat = self.decoder(z_q)

        return x_hat.float(), quantized_indices, z_q

    @torch.no_grad()
    def infer_nobf16(self, x):
        x = self.pad_audio(x)
        z_e = self.encoder(x)
        z_q, quantized_indices = self.quantizer.inference(z_e)
        x_hat = self.decoder(z_q)
        return x_hat.float(), quantized_indices, z_q

    @classmethod
    def from_pretrained(cls, repo_id="Ereboas/MagiCodec_16k_50hz", filename="MagiCodec-50Hz-Base.ckpt"):
        from huggingface_hub import hf_hub_download

        # Download the model file from Hugging Face Hub
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Load the model weights and initialize the class
        state_dict = torch.load(model_path, map_location="cpu")
        model = Generator()
        model.load_state_dict(state_dict)

        return model.eval()


if __name__ == "__main__":
    codec = Generator.from_pretrained().cuda()
    x = torch.randn(2, 16000).cuda()
    toks = codec.sig_to_toks(x)
    print(codec.toks_to_sig(toks).shape)
    print(toks.shape)