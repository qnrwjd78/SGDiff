import torch
import torch.nn as nn
from einops import rearrange


class NSGEncoder(nn.Module):
    """
    Neural Scene Graph encoder that turns object/relation triples into
    per-image local tokens for cross-attention.

    Inputs:
        graph = (objs, boxes, triples, obj_to_img, triple_to_img)
        - objs: (num_objs,) long
        - boxes: (num_objs, 4) float in [0,1]
        - triples: (num_triples, 3) long (s, p, o)
        - obj_to_img: (num_objs,) long
        - triple_to_img: (num_triples,) long

    Output:
        local_tokens: (B, max_tokens, dim) padded with zeros
    """

    def __init__(
        self,
        num_objs: int,
        num_preds: int,
        dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        max_tokens: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.dim = dim
        self.obj_embed = nn.Embedding(num_objs + 1, dim)
        self.pred_embed = nn.Embedding(num_preds + 1, dim)
        self.box_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, graph):
        objs, boxes, triples, obj_to_img, triple_to_img = graph
        device = objs.device

        obj_tokens = self.obj_embed(objs) + self.box_mlp(boxes)
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = s.squeeze(1), p.squeeze(1), o.squeeze(1)

        pred_tokens = self.pred_embed(p)
        so_tokens = torch.cat([obj_tokens[s], pred_tokens, obj_tokens[o]], dim=1)

        # pool triples per image, pad to max_tokens
        batch_size = int(obj_to_img.max().item()) + 1
        tokens = torch.zeros(batch_size, self.max_tokens, self.dim, device=device)
        mask = torch.zeros(batch_size, self.max_tokens, device=device, dtype=torch.bool)

        for img_idx in range(batch_size):
            rel_idx = (triple_to_img == img_idx).nonzero(as_tuple=False).squeeze(-1)
            cur_tokens = so_tokens[rel_idx]
            cur_len = min(cur_tokens.size(0), self.max_tokens)
            tokens[img_idx, :cur_len] = cur_tokens[:cur_len]
            mask[img_idx, :cur_len] = True

        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        return encoded

