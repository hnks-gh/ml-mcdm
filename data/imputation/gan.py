from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class GAINImputer(nn.Module):
    """
    Tier 3 Enhancement M-10: Generative Adversarial Imputation Networks (GAIN).
    
    Architecture for structured governance panel data.
    Captures MNAR selection mechanisms via adversarial training.
    Yoon, Jordon & van der Schaar (2018).
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        alpha: float = 100.0,
        hint_rate: float = 0.9,
        lr: float = 0.001,
        batch_size: int = 128,
        iterations: int = 3000,
        device: str = "cpu"
    ):
        super(GAINImputer, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha  # Reconstruction loss weight
        self.hint_rate = hint_rate
        self.lr = lr
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = torch.device(device)
        
        # Generator: (X_concat_M_concat_Z) -> X_hat
        # Input is x (dim) + mask (dim) + noise (dim)
        self.generator = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid() # Governance scores normalized [0, 1]
        ).to(self.device)
        
        # Discriminator: (X_hat_concat_H) -> M_hat (per-cell prob)
        # Input is x_hat (dim) + hint (dim)
        self.discriminator = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid()
        ).to(self.device)
        
    def _sample_noise(self, m, n):
        """Zero-mean noise sampled from uniform distribution."""
        return torch.FloatTensor(m, n).uniform_(0, 0.01).to(self.device)
        
    def _sample_hint(self, mask):
        """Generate hint (subset of true mask) - Yoon et al. Eq 4."""
        h_matrix = torch.rand_like(mask).to(self.device)
        h_mask = (h_matrix < self.hint_rate).float()
        return mask * h_mask + 0.5 * (1 - h_mask)
        
    def fit(self, data_array: np.ndarray) -> "GAINImputer":
        """Train the GAIN model on incomplete data."""
        # Preprocessing: Normalize [0, 1] for governance scores
        mask = (~np.isnan(data_array)).astype(float)
        data = np.nan_to_num(data_array, nan=0.0)
        
        # Fit normalization parameters
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        data_norm = (data - self.min_val) / (self.max_val - self.min_val + 1e-6)
        
        # Convert to torch
        data_torch = torch.from_numpy(data_norm).float().to(self.device)
        mask_torch = torch.from_numpy(mask).float().to(self.device)
        
        # Optimizers
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        # Training loop
        n = data_norm.shape[0]
        for it in range(self.iterations):
            # Sample batch indices
            idx = torch.randperm(n)[:self.batch_size]
            x_mb = data_torch[idx]
            m_mb = mask_torch[idx]
            
            # (1) Train Discriminator
            z_mb = self._sample_noise(x_mb.size(0), self.dim)
            h_mb = self._sample_hint(m_mb)
            
            # Generator output (imputed)
            # Yoon et al. Eq 1: x_tilde = m * x + (1-m) * g(x, m, z)
            gen_input = torch.cat([x_mb, m_mb, z_mb], dim=1)
            x_gen = self.generator(gen_input)
            x_tilde = m_mb * x_mb + (1 - m_mb) * x_gen
            
            # Discriminator loss: Binary Cross Entropy
            disc_input = torch.cat([x_tilde, h_mb], dim=1)
            m_hat = self.discriminator(disc_input)
            
            # Yoon et al. Eq 5
            disc_loss = -torch.mean(m_mb * torch.log(m_hat + 1e-8) + 
                                  (1 - m_mb) * torch.log(1 - m_hat + 1e-8))
            
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
            
            # (2) Train Generator
            # Sample again for stability
            z_mb = self._sample_noise(x_mb.size(0), self.dim)
            gen_input = torch.cat([x_mb, m_mb, z_mb], dim=1)
            x_gen = self.generator(gen_input)
            x_tilde = m_mb * x_mb + (1 - m_mb) * x_gen
            
            # Adversarial loss: trick discriminator into thinking imputed are real
            disc_input = torch.cat([x_tilde, h_mb], dim=1)
            m_hat = self.discriminator(disc_input)
            gen_adv_loss = -torch.mean((1 - m_mb) * torch.log(m_hat + 1e-8))
            
            # Reconstruction loss: force generator to reproduce observed values
            gen_mse_loss = torch.mean(m_mb * (x_mb - x_gen)**2) / torch.mean(m_mb + 1e-8)
            
            # Yoon et al. Eq 7-8: Total Gen loss
            gen_loss = gen_adv_loss + self.alpha * gen_mse_loss
            
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()
            
        return self
        
    def transform(self, data_array: np.ndarray) -> np.ndarray:
        """Impute missing data using trained GAIN model."""
        mask = (~np.isnan(data_array)).astype(float)
        data = np.nan_to_num(data_array, nan=0.0)
        data_norm = (data - self.min_val) / (self.max_val - self.min_val + 1e-6)
        
        # Convert to torch
        data_torch = torch.from_numpy(data_norm).float().to(self.device).detach()
        mask_torch = torch.from_numpy(mask).float().to(self.device).detach()
        
        # Generator input
        z_mb = torch.zeros(data_array.shape[0], self.dim).to(self.device)
        gen_input = torch.cat([data_torch, mask_torch, z_mb], dim=1)
        
        # Impute
        with torch.no_grad():
            x_gen = self.generator(gen_input)
            x_tilde = mask_torch * data_torch + (1 - mask_torch) * x_gen
            x_tilde_np = x_tilde.cpu().numpy()
            
        # Denormalize
        imputed_data = x_tilde_np * (self.max_val - self.min_val + 1e-6) + self.min_val
        return imputed_data
