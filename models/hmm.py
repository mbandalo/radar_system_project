import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneTransitionModel(nn.Module):
    """
    Hidden Markov Model / Dynamic Bayesian Network layer for scene transitions.
    Models P(scene_t | scene_{t-1}) with learnable transition matrix
    and emission probabilities via latent embeddings.
    """
    def __init__(self, num_states: int, embedding_dim: int):
        super().__init__()
        self.trans_logits = nn.Parameter(torch.zeros(num_states, num_states))
        self.emitter = nn.Linear(embedding_dim, num_states)

    def forward(self, prev_state_dist: torch.Tensor, latent_embed: torch.Tensor):
        """
        Compute posterior distribution over current scene state.

        Args:
            prev_state_dist: Tensor of shape (batch_size, num_states), prior P(scene_{t-1}) or posterior
            latent_embed: Tensor of shape (batch_size, embedding_dim), latent features aggregated from AE output
                          (e.g., mean of per-point embeddings for a scene)
        Returns:
            post_state_dist: Tensor of shape (batch_size, num_states), posterior P(scene_t | data)
        """
        # Compute log-transition probabilities
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        
        # prior over current state via marginalizing prev state: batch x states
        log_prior = torch.log(prev_state_dist + 1e-8) @ log_trans

        # emission log-probabilities from latent embedding
        emission_logits = self.emitter(latent_embed)  
        
        # Apply log_softmax over the num_states dimension (last dimension)
        log_emission = F.log_softmax(emission_logits, dim=-1)

        # Combine prior and emission to get unnormalized log-posterior
        log_post_unnorm = log_prior + log_emission
        
        # Normalize to get posterior distribution
        post_state_dist = F.softmax(log_post_unnorm, dim=1)
        return post_state_dist