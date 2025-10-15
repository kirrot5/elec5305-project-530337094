"""
Evaluation metrics for source separation
Author: Kiro Chen
"""

import numpy as np
from scipy import signal

def calculate_sdr(reference, estimated):
    """
    Calculate Signal-to-Distortion Ratio (SDR)

    Args:
        reference: Ground truth signal
        estimated: Estimated signal

    Returns:
        sdr: SDR value in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]

    # Calculate SDR
    numerator = np.sum(reference ** 2)
    denominator = np.sum((reference - estimated) ** 2)

    if denominator == 0:
        return np.inf

    sdr = 10 * np.log10(numerator / denominator)
    return sdr

def calculate_sir(reference, estimated, interference):
    """
    Calculate Signal-to-Interference Ratio (SIR)

    Args:
        reference: Target source (e.g., vocals)
        estimated: Estimated target
        interference: Interference source (e.g., accompaniment)

    Returns:
        sir: SIR value in dB
    """
    min_len = min(len(reference), len(estimated), len(interference))
    reference = reference[:min_len]
    estimated = estimated[:min_len]
    interference = interference[:min_len]

    # Project estimated onto reference
    projection = np.dot(estimated, reference) / (np.dot(reference, reference) + 1e-8)
    target_contribution = projection * reference

    # Interference in estimate
    interference_in_estimate = estimated - target_contribution

    numerator = np.sum(target_contribution ** 2)
    denominator = np.sum(interference_in_estimate ** 2)

    if denominator == 0:
        return np.inf

    sir = 10 * np.log10(numerator / denominator)
    return sir

def calculate_sar(reference, estimated):
    """
    Calculate Signal-to-Artifact Ratio (SAR)

    Args:
        reference: Ground truth signal
        estimated: Estimated signal

    Returns:
        sar: SAR value in dB
    """
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]

    # Artifacts = distortion
    error = estimated - reference

    numerator = np.sum(estimated ** 2)
    denominator = np.sum(error ** 2)

    if denominator == 0:
        return np.inf

    sar = 10 * np.log10(numerator / denominator)
    return sar

def evaluate_separation(reference, estimated, interference=None):
    """
    Comprehensive evaluation of separation quality

    Args:
        reference: Ground truth target (vocals)
        estimated: Estimated target
        interference: Ground truth interference (accompaniment)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {
        'SDR': calculate_sdr(reference, estimated),
        'SAR': calculate_sar(reference, estimated)
    }

    if interference is not None:
        metrics['SIR'] = calculate_sir(reference, estimated, interference)

    return metrics


# Testing
if __name__ == "__main__":
    # Generate test signals
    sr = 22050
    t = np.linspace(0, 1, sr)

    reference = np.sin(2 * np.pi * 440 * t)
    estimated = reference + 0.1 * np.random.randn(len(reference))

    metrics = evaluate_separation(reference, estimated)

    print("Evaluation Metrics Test:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f} dB")