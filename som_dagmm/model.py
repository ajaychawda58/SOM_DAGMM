"""Implements all the components of the DAGMM model."""

import torch
import numpy as np
from torch import nn
from minisom import MiniSom
from SOM import som_train


eps = torch.autograd.Variable(torch.FloatTensor([1.e-8]), requires_grad=False)

class SOM_DAGMM(nn.Module):
    def __init__(self, dagmm):
        super().__init__()
        self.dagmm = dagmm

    def forward(self, input):
        som = som_train(input)
        winners = [som.winner(i) for i in input]
        winners = torch.tensor([normalize_tuple(winners[i], 10) for i in range(len(winners))], dtype=torch.float32)
        return self.dagmm(input, winners)



        
        
class DAGMM(nn.Module):
    def __init__(self, compression_module, estimation_module, gmm_module):
        """
        Args:
            compression_module (nn.Module): an autoencoder model that
                implements at leat a function `self.encoder` to get the
                encoding of a given input.
            estimation_module (nn.Module): a FFNN model that estimates the
                memebership of each input to a each mixture of a GMM.
            gmm_module (nn.Module): a GMM model that implements its mixtures
                as a list of Mixture classes. The GMM model should implement
                the function `self._update_mixtures_parameters`.
        """
        super().__init__()

        self.compressor = compression_module
        self.estimator = estimation_module
        self.gmm = gmm_module
        

    def forward(self, input, winners):
        # Forward in the compression network.
        encoded = self.compressor.encode(input)
        decoded = self.compressor.decode(encoded)

        # Preparing the input for the estimation network.
        relative_ed = relative_euclidean_distance(input, decoded)
        cosine_sim = cosine_similarity(input, decoded)
        # Adding a dimension to prepare for concatenation.
        relative_ed = relative_ed.view(-1, 1)
        cosine_sim = relative_ed.view(-1, 1)
        latent_vectors = torch.cat([encoded, relative_ed, cosine_sim, winners], dim=1)
        # latent_vectors has shape [batch_size, dim_embedding + 2]

        # Updating the parameters of the mixture.
        if self.training:
            mixtures_affiliations = self.estimator(latent_vectors)
            # mixtures_affiliations has shape [batch_size, num_mixtures]
            self.gmm._update_mixtures_parameters(latent_vectors,
                                                 mixtures_affiliations)
        # Estimating the energy of the samples.
        return self.gmm(latent_vectors)


def relative_euclidean_distance(x1, x2, eps=eps):
    """x1 and x2 are assumed to be Variables or Tensors.
    They have shape [batch_size, dimension_embedding]"""
    num = torch.norm(x1 - x2, p=2, dim=1)  # dim [batch_size]
    denom = torch.norm(x1, p=2, dim=1)  # dim [batch_size]
    return num / torch.max(denom, eps)


def cosine_similarity(x1, x2, eps=eps):
    """x1 and x2 are assumed to be Variables or Tensors.
    They have shape [batch_size, dimension_embedding]"""
    dot_prod = torch.sum(x1 * x2, dim=1)  # dim [batch_size]
    dist_x1 = torch.norm(x1, p=2, dim=1)  # dim [batch_size]
    dist_x2 = torch.norm(x2, p=2, dim=1)  # dim [batch_size]
    return dot_prod / torch.max(dist_x1*dist_x2, eps)

def normalize_tuple(x, norm_val):
    a, b = x
    a = a/norm_val
    b = b/norm_val
    return (a,b)
