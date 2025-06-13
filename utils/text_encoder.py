import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, base_encoder, embed_dim):
        super().__init__()
        self.base_encoder = base_encoder
        self.projector = nn.Linear(base_encoder.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.base_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.projector(pooled)
