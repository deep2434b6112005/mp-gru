"""
model.py

Shared GestureClassifier model used by both
train_classifier.py and live_detect.py.
"""

import torch.nn as nn
import torch.nn.functional as F

from mp_gru import MPGRU


class GestureClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = 126,
        hidden_size: int = 64,
        embedding_size: int = 32,
        dropout: float = 0.3,
        **mpgru_kwargs,
    ):
        super().__init__()

        self.mpgru = MPGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            **mpgru_kwargs,
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Linear(hidden_size, embedding_size)

        # Auxiliary classifier
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, c, lengths):
        """
        Returns
        -------
        h_final : (B, hidden_size)
        embedding : (B, embedding_size)
        class_logits : (B, num_classes)
        """

        outputs, _, _ = self.mpgru(x, c)

        idx = (lengths - 1).clamp(min=0)
        idx = idx.view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
        h_final = outputs.gather(1, idx).squeeze(1)

        z = F.relu(self.fc1(h_final))
        z = self.dropout(z)

        embedding_raw = self.embedding(z)

        class_logits = self.classifier(embedding_raw)

        embedding = F.normalize(embedding_raw, p=2, dim=1)

        return h_final, embedding, class_logits
