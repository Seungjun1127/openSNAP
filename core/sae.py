# core/sae.py

import torch
import torch.nn as nn
import torch.optim as optim

class SparseAutoEncoder(nn.Module):
    def __init__(self, input_dimen, latent_dimen):

        # SAE initialization
        # input_dimen = number of activations
        # latent_dimen = dimension of the sparse latent space

        super(SparseAutoEncoder, self).__init__()

        # Encoder (Reduces dimensionality)
        self.encoder = nn.Sequential(
            nn.Linear(input_dimen, latent_dimen),
            nn.ReLU()
        )

        # Decoder (reconstructs input)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dimen, input_dimen)
        )

        # forward pass through the autoencoder
        def forward(self, x):                       # x = input tensor
            encoded = self.encoder(x)               # latent space representation
            decoded = self.decoder(encoded)
            return decoded                          # returns reconstructed input

# defining the loss fucntion

def sparse_autoencoder_loss(reconstructed, original, model, sparsity_weight = 1e-4):

    # reconstructed = reconstructed output from the decoder
    # original = original input data
    # sparsity_weight = weight of the sparsity penalty (L1)

    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    l1_penalty = 0
    for param in model.encoder.parameters():
        l1_penalty += torch.sum(torch.abs(param))

    total_loss = reconstruction_loss + sparsity_weight * l1_penalty
    