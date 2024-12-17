import torch
import torch.nn as nn
import torch.optim as optim

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmbeddingClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)
