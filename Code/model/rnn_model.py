from torch import nn
import torch

class GRU(nn.Module):
  def __init__(self, vocab_size, embed_dim, num_layers, hidden_size, output_size):
    super().__init__()

    self.num_layers = num_layers
    self.hidden_size = hidden_size

    self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    self.gru = nn.GRU(
        input_size=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True
    )
    self.classifier = nn.Linear(in_features=hidden_size, out_features=output_size)

  def forward(self, x):

    # Embedding and initial h0
    h0 = torch.zeros(size=(self.num_layers, x.shape[0], self.hidden_size))
    x_embed = self.embed(x) # -> (batch_size, sequence_length, hidden_size)

    # Pass through the model
    out, last_h = self.gru(x_embed, h0)

    # Max pool
    out, position = torch.max(out, dim=1)

    # classifier
    out = self.classifier(out)

    return out
