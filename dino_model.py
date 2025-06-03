import torch
import torch.nn as nn
import numpy as np


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