import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian_nll_loss(x_rec_mean: torch.Tensor, x_true: torch.Tensor, x_rec_logvar: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Calculates Gaussian negative log-likelihood loss."""
    recon_loss_per_feature = 0.5 * x_rec_logvar + 0.5 * ((x_rec_mean - x_true) ** 2) * torch.exp(-x_rec_logvar)
    recon_loss_per_point = recon_loss_per_feature.sum(dim=-1)

    if reduction == 'mean':
        return recon_loss_per_point.mean()
    elif reduction == 'sum':
        return recon_loss_per_point.sum()
    elif reduction == 'none':
        return recon_loss_per_point
    else:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

def kl_divergence(mu_latent: torch.Tensor, logvar_latent: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """Calculates KL divergence to a standard normal distribution."""
    kl_loss_per_point = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp(), dim=-1)
    kl_loss_mean = kl_loss_per_point.mean()

    if free_bits > 0.0:
        kl_loss_mean = torch.clamp(kl_loss_mean, min=free_bits)
    
    return kl_loss_mean

class FeatureExtractor(nn.Module):
    """MLP feature extractor for point-level encoding."""
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ProbabilisticAE(nn.Module):
    """
    Probabilistic Autoencoder: encoder outputs (mu, logvar), decoder reconstructs.
    Calculates reconstruction loss (Gaussian NLL) and KL divergence internally.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list, num_classes: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = FeatureExtractor(input_dim, hidden_dims)
        self.fc_mu = nn.Linear(self.encoder.output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.output_dim, latent_dim)

        dec_layers = []
        prev_dec_dim = latent_dim
        
        temp_hidden_dims_reversed = list(hidden_dims)
        temp_hidden_dims_reversed.reverse()

        if not temp_hidden_dims_reversed:
            decoder_output_for_head = latent_dim
        else:
            for h in temp_hidden_dims_reversed:
                dec_layers.append(nn.Linear(prev_dec_dim, h))
                dec_layers.append(nn.ReLU())
                prev_dec_dim = h
            decoder_output_for_head = prev_dec_dim

        self.decoder_base = nn.Sequential(*dec_layers)
        
        self.fc_recon_mean = nn.Linear(decoder_output_for_head, input_dim) 
        self.fc_recon_logvar = nn.Linear(decoder_output_for_head, input_dim) 

        self.num_classes = num_classes
        self.classifier = nn.Linear(latent_dim, self.num_classes) if self.num_classes is not None and self.num_classes > 0 else None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat_x = x.view(-1, self.input_dim) 

        features = self.encoder(flat_x)
        mu_latent = self.fc_mu(features)
        logvar_latent = self.fc_logvar(features)
        
        logvar_latent = torch.clamp(logvar_latent, min=-10.0, max=5.0) 
        
        z = self.reparameterize(mu_latent, logvar_latent)

        decoder_features = self.decoder_base(z)
        x_rec_mean = self.fc_recon_mean(decoder_features)
        x_rec_logvar = self.fc_recon_logvar(decoder_features)

        x_rec_logvar = torch.clamp(x_rec_logvar, min=-3.0, max=0.0)
        
        classification_logits = self.classifier(mu_latent) if self.classifier is not None else None
        
        if len(original_shape) == 3:
            num_points_padded = original_shape[1]
            
            x_rec_mean = x_rec_mean.view(original_shape[0], num_points_padded, self.input_dim)
            x_rec_logvar = x_rec_logvar.view(original_shape[0], num_points_padded, self.input_dim)
            mu_latent = mu_latent.view(original_shape[0], num_points_padded, self.latent_dim)
            logvar_latent = logvar_latent.view(original_shape[0], num_points_padded, self.latent_dim)
            
            if classification_logits is not None:
                classification_logits = classification_logits.view(original_shape[0], num_points_padded, self.num_classes)
        
        return x_rec_mean, mu_latent, logvar_latent, x_rec_logvar, classification_logits