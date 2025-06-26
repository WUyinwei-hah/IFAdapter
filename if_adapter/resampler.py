# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math
from torch import Tensor
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def default(v, d):
    return v if exists(v) else d

class GemmaRMSNormWithoutScale(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float())
        return output.type_as(x)
    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaMLPWithNorm(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        inner_dim = int(dim * mult)

        self.norm = GemmaRMSNormWithoutScale(inner_dim)
        self.gate_proj = nn.Linear(dim, inner_dim, bias=False)
        self.up_proj = nn.Linear(dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, dim, bias=False)
        self.act_fn = nn.GELU("tanh")

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj( self.norm(self.act_fn(self.gate_proj(x)).float() * self.up_proj(x)))

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

def generate_attention_map_mask(all_obj_attention_mask, latent_tokens_num):
    n, l = all_obj_attention_mask.shape
    eot_pos = all_obj_attention_mask.sum(axis=-1)-1
    all_obj_attention_mask[:, 0] = 0
    all_obj_attention_mask[torch.arange(n), eot_pos] = 0
    extended_tensor = torch.ones(n, latent_tokens_num).to(all_obj_attention_mask)
    key_mask = torch.cat([all_obj_attention_mask, extended_tensor], dim=-1).view(n, 1, -1)
    query_mask = torch.ones(n, latent_tokens_num).to(all_obj_attention_mask).view(n, -1, 1)
    attention_mask = torch.einsum("bij,bjk->bik", query_mask, key_mask)
    # attention_mask = torch.bmm(query_mask, key_mask)
    attention_mask = attention_mask.view(n, 1, latent_tokens_num, -1)

    return attention_mask

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.q_norm = GemmaRMSNorm(dim_head)
        self.k_norm = GemmaRMSNorm(dim_head)

    def forward(self, x, latents, all_obj_attention_mask):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n_layers, l1, d)
            latent (torch.Tensor): latent features
                shape (b, n_layers, l2, d)
            all_obj_attention_mask:

        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, n, l, dim = latents.shape
        latents = latents.reshape(b*n, -1, dim)
        x = x.reshape(b*n, -1, dim)
        all_obj_attention_mask = all_obj_attention_mask.reshape(b*n, -1) # only attend to the tokens except eot, sot, padding

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        q, k = self.q_norm(q), self.k_norm(k)

        attention_mask = generate_attention_map_mask(all_obj_attention_mask, q.shape[2])

        # attention
        out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask.to(torch.bool)
            )

        out = out.permute(0, 2, 1, 3).reshape(b, n, l, -1)

        return self.to_out(out)

class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 2, 3, 1).reshape(x.shape[0], -1)


class PositionNet(nn.Module):
    def __init__(self, out_dim, fourier_freqs=64):
        super().__init__()
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        # -------------------------------------------------------------- #
        self.linears_position = nn.Sequential(
            nn.Linear(self.position_dim, 768),
            nn.SiLU(),
            nn.Linear(768, 768),
            nn.SiLU(),
            nn.Linear(768, out_dim),
        )

    def forward(self, boxes):

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*1*4 --> B*1*C torch.Size([5, 1, 64])
        xyxy_embedding = self.linears_position(xyxy_embedding)  # B*1*C --> B*1*768 torch.Size([5, 1, 768])

        return xyxy_embedding

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=160,
        heads=16,
        num_queries=8,
        output_dim=1024,
        phrase_embeddings_dim=2048,
        ff_mult=4,
        fourier_freqs=256,
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs) # fourier_freqs * 2 * 4 = 2048
        # self.pos_net = PositionNet(out_dim=phrase_embeddings_dim, fourier_freqs=8)
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5) # learnable latents
        self.proj_in = nn.Linear(phrase_embeddings_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])


        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        # FeedForward(dim=dim, mult=ff_mult),
                        GemmaMLPWithNorm(dim=dim, mult=ff_mult),
                        
                    ]
                )
            )

    def forward(self, x, all_obj_attention_mask):
        """
        x: total_phrases_in_one_batch, extract_embedding_layers, l, dim1+dim2
        """

        # self.latents: 1 3 2048 -> B numlay 3 2048
        latents = self.latents.unsqueeze(0).repeat(x.size(0),x.size(1), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents, all_obj_attention_mask) + latents
            latents = ff(latents) + latents

        latents = self.attention_norm(latents)
        latents = self.proj_out(latents)
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)

if __name__ == "__main__":
    pos_net = PositionNet(2048, fourier_freqs=64)
    # fourier_embedder = FourierEmbedder()
    boxes = torch.tensor([[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]])
    result = pos_net(boxes.reshape(2, -1))