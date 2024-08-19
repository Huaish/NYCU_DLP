import torch.nn as nn
from diffusers import UNet2DModel

class ConditionalDDPM(nn.Module):
    def __init__(self, num_labels=24, dim=512):
        super().__init__()
        self.label_embedding = nn.Linear(num_labels, dim)
        self.diffusion = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(dim//4, dim//4, dim//2, dim//2, dim, dim),
            down_block_types=[
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ],
            up_block_types=[
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
            class_embed_type="identity",
            num_class_embeds=dim
        )
        
        
    def forward(self, x, timesteps, label):
        label = self.label_embedding(label)
        return self.diffusion(x, timesteps, label).sample