# models/classifier.py
"""
Lightweight CNN classifier for MNIST / FashionMNIST.
Used by eval.py to:
  1. Extract intermediate features for Dataset-FID / KID computation.
  2. Evaluate classifier accuracy and class-distribution entropy on generated samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    LeNet-style CNN.
    Feature layer: the penultimate (128-dim) representation.
    """

    def __init__(self, in_ch=1, num_classes=10, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),  # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1 (global avg pool)
        )
        self.fc_feat = nn.Linear(128, feature_dim)
        self.fc_out = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        z = self.conv(x).flatten(1)  # (B, 128)
        feat = F.relu(self.fc_feat(z))  # (B, feature_dim)
        logits = self.fc_out(feat)
        return logits

    @torch.no_grad()
    def extract_features(self, x):
        """Return the feature-layer activations (B, feature_dim)."""
        z = self.conv(x).flatten(1)
        return F.relu(self.fc_feat(z))


def train_classifier(dataset, num_classes=10, epochs=10, device="cpu"):
    """
    Train a LeNet classifier on `dataset` and return the trained model.

    Args:
        dataset    : a torch Dataset returning (image, label) in [-1, 1]
        num_classes: number of output classes
        epochs     : training epochs
        device     : torch device

    Returns:
        model: trained LeNet (in eval mode)
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, num_workers=2
    )
    model = LeNet(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        print(
            f"  [Classifier] epoch {epoch + 1}/{epochs}  "
            f"loss={total_loss / total:.4f}  acc={correct / total:.4f}"
        )

    model.eval()
    return model
