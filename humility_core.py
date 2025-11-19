"""
Humility Protocol - Core Implementation
=======================================

Multi-AI Collaborative Framework for Uncertainty-Calibrated Intelligence

Authors: Joshua A. Duran, Claude (Anthropic), GPT-5 (OpenAI), Grok (xAI)
License: MIT
Repository: https://github.com/ATHENANOUSMACHINA/humility-protocol
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import deque


class HumilityLayer(nn.Module):
    """
    Core humility layer that transforms model outputs into calibrated predictions.
    
    Implements the complete Humility Protocol:
    - Uncertainty Estimation Module (UEM)
    - Humility Calibration Layer (HCL)
    - Confidence Modulation Module (CMM)
    - Metacognitive Feedback Loop (MFL)
    
    Args:
        input_dim: Dimension of model outputs
        n_estimators: Number of ensemble members for epistemic uncertainty
        H_min: Minimum humility bound (default: 0.15)
        H_max: Maximum humility bound (default: 0.85)
        history_size: Size of metacognitive history buffer
    """
    
    def __init__(
        self,
        input_dim: int,
        n_estimators: int = 3,
        H_min: float = 0.15,
        H_max: float = 0.85,
        history_size: int = 1000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.H_min = H_min
        self.H_max = H_max
        
        # Uncertainty Estimation Module (UEM)
        self.epistemic_estimator = EpistemicUncertaintyHead(input_dim, n_estimators)
        self.aleatoric_estimator = AleatoricUncertaintyHead(input_dim)
        self.metacognitive_monitor = MetacognitiveMonitor(input_dim, history_size)
        
        # Humility Calibration Layer (HCL)
        self.humility_calibration = HumilityCalibrationNetwork(H_min, H_max)
        
        # Learnable temperature scaling base
        self.base_temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass computing calibrated outputs with humility.
        
        Args:
            x: Input logits [batch_size, n_classes]
            return_components: If True, return uncertainty components
            
        Returns:
            calibrated_logits: Temperature-scaled logits
            H: Humility coefficient for each sample
            components: Optional dict of uncertainty components
        """
        
        # 1. Uncertainty Estimation (UEM)
        epistemic = self.epistemic_estimator(x)
        aleatoric = self.aleatoric_estimator(x)
        metacognitive = self.metacognitive_monitor(x)
        
        # 2. Humility Calibration (HCL)
        H = self.humility_calibration(epistemic, aleatoric, metacognitive)
        
        # 3. Confidence Modulation (CMM)
        calibrated_logits = self.apply_humility_scaling(x, H)
        
        if return_components:
            components = {
                'epistemic': epistemic,
                'aleatoric': aleatoric,
                'metacognitive': metacognitive,
                'H': H
            }
            return calibrated_logits, H, components
        
        return calibrated_logits, H, None
    
    def apply_humility_scaling(
        self,
        logits: torch.Tensor,
        H: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temperature scaling based on humility coefficient.
        Higher humility → higher temperature → lower confidence.
        
        Args:
            logits: Raw model outputs
            H: Humility coefficients
            
        Returns:
            Temperature-scaled logits
        """
        # Temperature: T = base_temp * (1 + 2*H)
        # This gives T ∈ [base_temp, 3*base_temp]
        temperature = self.base_temperature * (1 + 2 * H.unsqueeze(1))
        calibrated = logits / temperature
        return calibrated
    
    def update_metacognitive_history(
        self,
        x: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        H: torch.Tensor
    ):
        """
        Update the Metacognitive Feedback Loop with actual performance.
        
        Args:
            x: Input features
            predictions: Model predictions
            labels: Ground truth labels
            H: Predicted humility coefficients
        """
        self.metacognitive_monitor.record_batch(x, predictions, labels, H)


class EpistemicUncertaintyHead(nn.Module):
    """
    Estimates epistemic uncertainty via ensemble disagreement.
    Uses multiple estimator heads and measures variance.
    """
    
    def __init__(self, input_dim: int, n_estimators: int = 3):
        super().__init__()
        self.estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(n_estimators)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute epistemic uncertainty as ensemble variance."""
        estimates = torch.stack([est(x) for est in self.estimators])
        epistemic_uncertainty = torch.var(estimates, dim=0).squeeze()
        return epistemic_uncertainty


class AleatoricUncertaintyHead(nn.Module):
    """
    Estimates aleatoric (irreducible) uncertainty.
    Learns to predict inherent randomness in the data.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute aleatoric uncertainty."""
        return self.uncertainty_net(x).squeeze()


class MetacognitiveMonitor:
    """
    Tracks the system's awareness of its own limitations.
    Maintains history of predictions and learns to predict errors.
    """
    
    def __init__(self, input_dim: int, capacity: int = 1000):
        self.capacity = capacity
        self.history = deque(maxlen=capacity)
        
        # Simple nearest-neighbor for context similarity
        self.context_buffer = deque(maxlen=capacity)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate metacognitive uncertainty based on historical performance
        in similar contexts.
        """
        if len(self.history) < 10:
            # Not enough history, return moderate uncertainty
            return torch.ones(x.size(0), device=x.device) * 0.5
        
        # Compute similarity to historical contexts
        similarities = self._compute_context_similarity(x)
        
        # Get historical accuracy for similar contexts
        historical_accuracy = self._get_historical_accuracy(similarities)
        
        # High uncertainty when historical accuracy is low
        metacognitive = 1.0 - historical_accuracy
        return torch.tensor(metacognitive, device=x.device)
    
    def _compute_context_similarity(self, x: torch.Tensor) -> np.ndarray:
        """Compute cosine similarity to historical contexts."""
        if len(self.context_buffer) == 0:
            return np.ones(x.size(0))
        
        # Simplified: just return uniform similarity for now
        # In production, use proper similarity computation
        return np.ones(x.size(0)) * 0.5
    
    def _get_historical_accuracy(self, similarities: np.ndarray) -> float:
        """Get average accuracy for similar historical contexts."""
        if len(self.history) == 0:
            return 0.5
        
        accuracies = [entry['correct'] for entry in self.history]
        return np.mean(accuracies)
    
    def record_batch(
        self,
        x: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        H: torch.Tensor
    ):
        """Record a batch of predictions for metacognitive learning."""
        correct = (predictions.argmax(dim=-1) == labels).float()
        
        for i in range(x.size(0)):
            self.history.append({
                'H': H[i].item(),
                'correct': correct[i].item()
            })
            self.context_buffer.append(x[i].detach().cpu().numpy())


class HumilityCalibrationNetwork(nn.Module):
    """
    Maps uncertainty components to unified humility coefficient.
    Learns the optimal combination of epistemic, aleatoric, and
    metacognitive uncertainty.
    """
    
    def __init__(self, H_min: float = 0.15, H_max: float = 0.85):
        super().__init__()
        
        self.H_min = H_min
        self.H_max = H_max
        
        self.calibration_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
        metacognitive: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine uncertainty components into humility coefficient.
        
        Args:
            epistemic: Epistemic uncertainty [batch_size]
            aleatoric: Aleatoric uncertainty [batch_size]
            metacognitive: Metacognitive uncertainty [batch_size]
            
        Returns:
            H: Humility coefficients in [H_min, H_max]
        """
        # Stack uncertainties
        uncertainties = torch.stack([
            epistemic,
            aleatoric,
            metacognitive
        ], dim=-1)
        
        # Map to [0, 1]
        raw_H = self.calibration_net(uncertainties).squeeze()
        
        # Scale to [H_min, H_max]
        H = self.H_min + (self.H_max - self.H_min) * raw_H
        
        return H


def humility_aware_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    H: torch.Tensor,
    lambda_calibration: float = 0.3,
    lambda_uncertainty: float = 0.2,
    lambda_metacognitive: float = 0.1
) -> torch.Tensor:
    """
    Combined loss function optimizing for both accuracy and calibration.
    
    Args:
        logits: Model outputs
        labels: Ground truth
        H: Humility coefficients
        lambda_*: Loss weights
        
    Returns:
        Combined loss
    """
    # 1. Task loss (standard cross-entropy)
    task_loss = F.cross_entropy(logits, labels)
    
    # 2. Calibration loss (penalize confident mistakes)
    predictions = logits.argmax(dim=-1)
    confidence = 1 - H
    mistakes = (predictions != labels).float()
    calibration_loss = (mistakes * confidence.pow(2)).mean()
    
    # 3. Uncertainty quality loss (align H with errors)
    uncertainty_loss = F.mse_loss(H, mistakes)
    
    # 4. Prevent pathological uncertainty
    paralysis_penalty = F.relu(H - 0.8).mean()
    
    # Combined
    total_loss = (
        task_loss +
        lambda_calibration * calibration_loss +
        lambda_uncertainty * uncertainty_loss +
        lambda_metacognitive * paralysis_penalty
    )
    
    return total_loss


# ============================================================================
# METRICS
# ============================================================================

def compute_ece(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        predictions: Predicted classes
        confidences: Prediction confidences
        labels: Ground truth labels
        n_bins: Number of calibration bins
        
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        
        if mask.sum() > 0:
            bin_acc = (predictions[mask] == labels[mask]).mean()
            bin_conf = confidences[mask].mean()
            bin_size = mask.mean()
            
            ece += bin_size * abs(bin_acc - bin_conf)
    
    return ece


def compute_opi(ece: float, accuracy: float) -> float:
    """
    Compute Overconfidence Pathology Index.
    
    OPI = ECE / Accuracy
    
    Lower is better. OPI < 0.05 indicates excellent calibration.
    """
    if accuracy == 0:
        return float('inf')
    return ece / accuracy


def compute_ohr(H_ood: np.ndarray, H_train: np.ndarray) -> float:
    """
    Compute Out-of-Distribution Humility Ratio.
    
    OHR = mean(H_ood) / mean(H_train)
    
    Should be > 1.3 (at least 30% higher for OOD).
    """
    return H_ood.mean() / H_train.mean()


if __name__ == "__main__":
    # Quick test
    print("Humility Protocol - Core Library")
    print("=" * 50)
    
    # Create a simple test
    batch_size = 32
    n_classes = 10
    
    # Create humility layer
    humility_layer = HumilityLayer(input_dim=n_classes, n_estimators=3)
    
    # Random logits
    logits = torch.randn(batch_size, n_classes)
    labels = torch.randint(0, n_classes, (batch_size,))
    
    # Forward pass
    calibrated_logits, H, components = humility_layer(logits, return_components=True)
    
    # Compute loss
    loss = humility_aware_loss(calibrated_logits, labels, H)
    
    print(f"Input shape: {logits.shape}")
    print(f"Humility coefficients: {H}")
    print(f"Mean H: {H.mean():.3f}")
    print(f"Loss: {loss.item():.4f}")
    print("\n✓ Core library functional!")
