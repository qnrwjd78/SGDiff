import torch
import torch.nn as nn
import clip


class CLIPGlobalEncoder(nn.Module):
    """
    Frozen CLIP image encoder that produces a global style/context embedding.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda", normalize: bool = True):
        super().__init__()
        model, _ = clip.load(model_name, device=device, jit=False)
        self.visual = model.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.output_dim = self.visual.output_dim

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        # image assumed in [-1,1], convert to CLIP preprocessing: [0,1], then normalize
        x = (image + 1.0) / 2.0
        # CLIP expects NCHW float32 on device
        x = x.to(next(self.visual.parameters()).device, dtype=torch.float32)
        feats = self.visual(x)
        if self.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats


class FiLMGenerator(nn.Module):
    """
    Generates (gamma, beta) for FiLM modulation at different channel widths.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict()

    def register_channel(self, channels: int):
        key = str(channels)
        if key not in self.heads:
            self.heads[key] = nn.Linear(self.hidden_dim, channels * 2)

    def forward(self, global_embedding: torch.Tensor, channels: int):
        key = str(channels)
        if key not in self.heads:
            # allow lazy registration for unexpected channel sizes (e.g., skip connections)
            self.register_channel(channels)
        h = self.hidden(global_embedding)
        film = self.heads[key](h)
        gamma, beta = film.chunk(2, dim=-1)
        return gamma, beta
