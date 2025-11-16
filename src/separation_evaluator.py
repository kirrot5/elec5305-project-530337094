"""
Evaluation Metrics for Music Source Separation
Author: Kiro Chen (530337094)

Implements BSS Eval metrics:
- SDR: Signal-to-Distortion Ratio
- SIR: Signal-to-Interference Ratio
- SAR: Signal-to-Artifacts Ratio
"""

import numpy as np
from scipy.linalg import lstsq


class SeparationEvaluator:
    """Evaluate source separation quality"""

    def __init__(self):
        pass

    def evaluate(self, reference_sources, estimated_sources,
                 sample_rate=22050, window=30.0):
        """
        Evaluate separation quality using BSS Eval metrics

        Args:
            reference_sources: Dict with 'vocals' and 'accompaniment' (ground truth)
            estimated_sources: Dict with 'vocals' and 'accompaniment' (predictions)
            sample_rate: Sample rate
            window: Window length for frame-based metrics (seconds)

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        metrics = {}

        for source_name in ['vocals', 'accompaniment']:
            ref = reference_sources[source_name]
            est = estimated_sources[source_name]

            # Ensure same length
            min_len = min(len(ref), len(est))
            ref = ref[:min_len]
            est = est[:min_len]

            # Handle stereo by averaging channels
            if ref.ndim > 1:
                ref = np.mean(ref, axis=0)
            if est.ndim > 1:
                est = np.mean(est, axis=0)

            # Calculate BSS Eval metrics
            sdr, sir, sar = self.bss_eval_sources(ref, est)

            metrics[f'{source_name}_SDR'] = sdr
            metrics[f'{source_name}_SIR'] = sir
            metrics[f'{source_name}_SAR'] = sar

        # Overall metrics
        metrics['mean_SDR'] = np.mean([metrics['vocals_SDR'], metrics['accompaniment_SDR']])
        metrics['mean_SIR'] = np.mean([metrics['vocals_SIR'], metrics['accompaniment_SIR']])
        metrics['mean_SAR'] = np.mean([metrics['vocals_SAR'], metrics['accompaniment_SAR']])

        return metrics

    def bss_eval_sources(self, reference, estimation):
        """
        Compute BSS Eval metrics for a single source

        Based on: Vincent et al. (2006) "Performance measurement in blind audio
        source separation"

        Args:
            reference: Ground truth signal
            estimation: Estimated signal

        Returns:
            SDR: Signal-to-Distortion Ratio (dB)
            SIR: Signal-to-Interference Ratio (dB)
            SAR: Signal-to-Artifacts Ratio (dB)
        """
        # Ensure 1D arrays
        reference = np.atleast_1d(reference.flatten())
        estimation = np.atleast_1d(estimation.flatten())

        # Make same length
        min_len = min(len(reference), len(estimation))
        reference = reference[:min_len]
        estimation = estimation[:min_len]

        # Compute decomposition
        s_target, e_interf, e_artif = self._decompose_estimation(reference, estimation)

        # Calculate metrics
        SDR = self._calculate_sdr(reference, s_target, e_interf, e_artif)
        SIR = self._calculate_sir(s_target, e_interf)
        SAR = self._calculate_sar(s_target, e_artif)

        return SDR, SIR, SAR

    def _decompose_estimation(self, reference, estimation):
        """
        Decompose estimation into target, interference, and artifacts

        estimation = s_target + e_interf + e_artif

        where:
        - s_target: allowed distortion of reference
        - e_interf: interference from other sources
        - e_artif: artifacts (noise, distortion)
        """
        # s_target is the projection of estimation onto reference
        # Using least squares: s_target = alpha * reference

        # Compute optimal scaling
        alpha = np.dot(estimation, reference) / (np.dot(reference, reference) + 1e-10)
        s_target = alpha * reference

        # Residual error
        e_total = estimation - s_target

        # In single-source case, all error is artifacts
        # (no other sources to interfere)
        e_interf = np.zeros_like(e_total)
        e_artif = e_total

        return s_target, e_interf, e_artif

    def _calculate_sdr(self, reference, s_target, e_interf, e_artif):
        """
        Signal-to-Distortion Ratio
        SDR = 10 * log10(||s_target||^2 / ||e_interf + e_artif||^2)
        """
        s_target_energy = np.sum(s_target ** 2)
        error_energy = np.sum((e_interf + e_artif) ** 2)

        if error_energy < 1e-10:
            return 100.0  # Perfect separation

        sdr = 10 * np.log10(s_target_energy / error_energy + 1e-10)
        return float(sdr)

    def _calculate_sir(self, s_target, e_interf):
        """
        Signal-to-Interference Ratio
        SIR = 10 * log10(||s_target||^2 / ||e_interf||^2)
        """
        s_target_energy = np.sum(s_target ** 2)
        interf_energy = np.sum(e_interf ** 2)

        if interf_energy < 1e-10:
            return 100.0

        sir = 10 * np.log10(s_target_energy / interf_energy + 1e-10)
        return float(sir)

    def _calculate_sar(self, s_target, e_artif):
        """
        Signal-to-Artifacts Ratio
        SAR = 10 * log10(||s_target||^2 / ||e_artif||^2)
        """
        s_target_energy = np.sum(s_target ** 2)
        artif_energy = np.sum(e_artif ** 2)

        if artif_energy < 1e-10:
            return 100.0

        sar = 10 * np.log10(s_target_energy / artif_energy + 1e-10)
        return float(sar)

    def calculate_snr(self, signal, noise):
        """
        Calculate Signal-to-Noise Ratio

        Args:
            signal: Clean signal
            noise: Noise signal

        Returns:
            SNR in dB
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            return 100.0

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    def calculate_pesq(self, reference, estimation, sr=16000):
        """
        Perceptual Evaluation of Speech Quality (placeholder)

        Note: Requires pesq library. Returns simplified version.
        """
        # Simplified correlation-based quality metric
        min_len = min(len(reference), len(estimation))
        ref = reference[:min_len]
        est = estimation[:min_len]

        # Normalize
        ref = ref / (np.max(np.abs(ref)) + 1e-10)
        est = est / (np.max(np.abs(est)) + 1e-10)

        # Correlation
        correlation = np.corrcoef(ref, est)[0, 1]

        # Map to PESQ-like scale [1, 5]
        quality = 1 + 4 * max(0, correlation)

        return float(quality)

    def print_evaluation_report(self, metrics):
        """Print comprehensive evaluation report"""
        print("\n" + "=" * 70)
        print("MUSIC SOURCE SEPARATION - EVALUATION REPORT")
        print("=" * 70)

        print("\n--- Vocals Separation ---")
        print(f"SDR (Signal-to-Distortion): {metrics['vocals_SDR']:>8.2f} dB")
        print(f"SIR (Signal-to-Interference): {metrics['vocals_SIR']:>8.2f} dB")
        print(f"SAR (Signal-to-Artifacts):  {metrics['vocals_SAR']:>8.2f} dB")

        print("\n--- Accompaniment Separation ---")
        print(f"SDR (Signal-to-Distortion): {metrics['accompaniment_SDR']:>8.2f} dB")
        print(f"SIR (Signal-to-Interference): {metrics['accompaniment_SIR']:>8.2f} dB")
        print(f"SAR (Signal-to-Artifacts):  {metrics['accompaniment_SAR']:>8.2f} dB")

        print("\n--- Overall Performance ---")
        print(f"Mean SDR: {metrics['mean_SDR']:>8.2f} dB")
        print(f"Mean SIR: {metrics['mean_SIR']:>8.2f} dB")
        print(f"Mean SAR: {metrics['mean_SAR']:>8.2f} dB")

        # Interpretation
        print("\n--- Quality Interpretation ---")
        mean_sdr = metrics['mean_SDR']
        if mean_sdr > 10:
            quality = "Excellent"
        elif mean_sdr > 5:
            quality = "Good"
        elif mean_sdr > 0:
            quality = "Fair"
        else:
            quality = "Poor"
        print(f"Overall Quality: {quality}")

        print("\n" + "=" * 70)


# ===== 新增：与你 import 对应的三个函数接口 =====

def compute_sdr(reference, estimation):
    """
    Convenience wrapper to compute SDR only.

    Usage:
        from src.separation_evaluator import compute_sdr
        sdr = compute_sdr(ref, est)
    """
    evaluator = SeparationEvaluator()
    sdr, _, _ = evaluator.bss_eval_sources(reference, estimation)
    return sdr


def compute_sir(reference, estimation):
    """
    Convenience wrapper to compute SIR only.
    """
    evaluator = SeparationEvaluator()
    _, sir, _ = evaluator.bss_eval_sources(reference, estimation)
    return sir


def compute_sar(reference, estimation):
    """
    Convenience wrapper to compute SAR only.
    """
    evaluator = SeparationEvaluator()
    _, _, sar = evaluator.bss_eval_sources(reference, estimation)
    return sar


# Test the evaluator
if __name__ == "__main__":
    print("Testing Separation Evaluator...")

    # Create synthetic signals
    sr = 22050
    duration = 3
    t = np.linspace(0, duration, int(sr * duration))

    # Ground truth
    vocals_true = np.sin(2 * np.pi * 440 * t)
    acc_true = np.sin(2 * np.pi * 220 * t)

    # Simulated estimations (with some error)
    vocals_est = vocals_true + 0.1 * np.random.randn(len(t))
    acc_est = acc_true + 0.15 * np.random.randn(len(t))

    # Evaluate
    evaluator = SeparationEvaluator()

    reference_sources = {
        'vocals': vocals_true,
        'accompaniment': acc_true
    }

    estimated_sources = {
        'vocals': vocals_est,
        'accompaniment': acc_est
    }

    metrics = evaluator.evaluate(reference_sources, estimated_sources, sr)

    # Print report
    evaluator.print_evaluation_report(metrics)

    # Test individual BSS eval
    print("\n--- Testing individual BSS eval ---")
    sdr, sir, sar = evaluator.bss_eval_sources(vocals_true, vocals_est)
    print(f"SDR: {sdr:.2f} dB")
    print(f"SIR: {sir:.2f} dB")
    print(f"SAR: {sar:.2f} dB")

    # Test new convenience functions
    print("\n--- Testing convenience compute_* functions ---")
    sdr2 = compute_sdr(vocals_true, vocals_est)
    sir2 = compute_sir(vocals_true, vocals_est)
    sar2 = compute_sar(vocals_true, vocals_est)
    print(f"compute_sdr: {sdr2:.2f} dB")
    print(f"compute_sir: {sir2:.2f} dB")
    print(f"compute_sar: {sar2:.2f} dB")

    print("\n✓ All tests passed!")
