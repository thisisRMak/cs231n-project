import torch
import torch.nn as nn
import numpy as np

from einops import rearrange


# class DINOImageClassifier(nn.Module):
#     def __init__(self, input_dim=384, hidden_dim=768, num_classes=18):
#         super(DINOImageClassifier, self).__init__()
        
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         """
#         Args:
#             x (torch.Tensor): DINO features of shape [batch_size, 3600, 384]

#         Returns:
#             logits (torch.Tensor): Class scores of shape [batch_size, 18]
#         """
        
#         x_pooled = x.mean(dim=1)  # [batch_size, 384]
#         logits = self.classifier(x_pooled)  # [batch_size, 18]
#         return logits


## Version 1
## From DINO: 3600,364
## Fwd: 3600,18 -> 18

class DINOImageClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=768, num_classes=18):
        super(DINOImageClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, "Token count must be a perfect square"

        x = x.permute(0, 2, 1).contiguous()  # [B, D, N]
        x = x.view(B, D, H, W)               # [B, D, H, W]
        logits = self.model(x)
        return logits
    
## Version 2
## From DINO: N,3600,364
## Our forward pass: N*3600,364-> Nx3600,2

class DINOPatchClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=768, num_classes=2):
        super(DINOPatchClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for patch-wise classification.
        
        Args:
            x: Input tensor of shape (N*P, D) where:
               N = batch size
               P = number of patches per image (3600)
               D = feature dimension (384)
        
        Returns:
            Tensor of shape (N*P, C) where C is number of classes
        """
        # Input should be flattened: (N*P, D)
        assert len(x.shape) == 2, f"Expected 2D input (N*P, D), got shape {x.shape}"
        assert x.shape[1] == 384, f"Expected feature dim 384, got {x.shape[1]}"
        
        logits = self.model(x)
        return logits