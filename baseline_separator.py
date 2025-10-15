"""
Baseline vocal separation using STFT and Ideal Binary Mask
"""

import numpy as np

import librosa
from scipy import signal


class STFTSeparator:
    """STFT-based baseline separator using spectral masking"""

    def __init__(self, n_fft=2048, hop_length=512, threshold=0.5):
        """
        Args:
            n_fft: FFT window size
            hop_length: Hop length for STFT
            threshold: Threshold for binary masking (0-1)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.threshold = threshold

    def compute_stft(self, audio):
        """Compute Short-Time Fourier Transform"""
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        return magnitude, phase

    def ideal_binary_mask(self, mixture_mag, vocals_mag):
        """
        Compute Ideal Binary Mask (IBM)
        IBM(t,f) = 1 if vocals_mag > accompaniment_mag, else 0

        Args:
            mixture_mag: Magnitude spectrogram of mixture
            vocals_mag: Magnitude spectrogram of clean vocals (ground truth)

        Returns:
            mask: Binary mask
        """
        # Accompaniment magnitude = mixture - vocals
        accompaniment_mag = np.abs(mixture_mag - vocals_mag)

        # Create binary mask: 1 where vocals dominate
        mask = (vocals_mag > accompaniment_mag).astype(float)

        return mask

    def ideal_ratio_mask(self, mixture_mag, vocals_mag, alpha=1.0):
        """
        Compute Ideal Ratio Mask (IRM)
        IRM(t,f) = (vocals_mag / mixture_mag)^alpha

        Args:
            mixture_mag: Magnitude spectrogram of mixture
            vocals_mag: Magnitude spectrogram of clean vocals
            alpha: Exponent parameter (default=1.0)

        Returns:
            mask: Ratio mask
        """
        # Avoid division by zero
        eps = 1e-8
        mask = np.power(
            vocals_mag / (mixture_mag + eps),
            alpha
        )

        # Clip to [0, 1]
        mask = np.clip(mask, 0, 1)

        return mask

    def separate(self, mixture_audio, vocals_audio=None, mask_type='ibm'):
        """
        Separate vocals from mixture

        Args:
            mixture_audio: Mixed audio signal
            vocals_audio: Clean vocals (for supervised masking)
            mask_type: 'ibm' (Ideal Binary Mask) or 'irm' (Ideal Ratio Mask)

        Returns:
            separated_vocals: Separated vocal audio
            mask: The computed mask
        """
        # Compute STFT
        mixture_mag, mixture_phase = self.compute_stft(mixture_audio)

        if vocals_audio is not None:
            # Supervised separation with ground truth
            vocals_mag, _ = self.compute_stft(vocals_audio)

            if mask_type == 'ibm':
                mask = self.ideal_binary_mask(mixture_mag, vocals_mag)
            elif mask_type == 'irm':
                mask = self.ideal_ratio_mask(mixture_mag, vocals_mag)
            else:
                raise ValueError(f"Unknown mask type: {mask_type}")
        else:
            # Unsupervised separation (simple energy-based mask)
            # This is a naive approach - vocals often in mid frequencies
            mask = self._simple_vocal_mask(mixture_mag)

        # Apply mask to mixture
        separated_mag = mixture_mag * mask

        # Reconstruct audio using mixture phase
        separated_stft = separated_mag * np.exp(1j * mixture_phase)
        separated_audio = librosa.istft(
            separated_stft,
            hop_length=self.hop_length
        )

        return separated_audio, mask

    def _simple_vocal_mask(self, magnitude):
        """
        Simple unsupervised vocal mask based on frequency characteristics
        Vocals typically dominant in 200-3000 Hz range
        """
        freq_bins = magnitude.shape[0]
        mask = np.ones_like(magnitude)

        # Emphasize mid-frequency range (simplified approach)
        # This is not very effective but serves as a baseline
        vocal_start = int(freq_bins * 0.1)  # ~200 Hz at 22050 sr
        vocal_end = int(freq_bins * 0.6)  # ~3000 Hz

        # Create a simple frequency-based mask
        mask[:vocal_start] *= 0.3
        mask[vocal_end:] *= 0.3

        return mask


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create separator
    separator = STFTSeparator(n_fft=2048, hop_length=512)

    # Generate test signal
    sr = 22050
    duration = 2
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate mixture: vocal (440 Hz) + accompaniment (220 Hz)
    vocals = np.sin(2 * np.pi * 440 * t)
    accompaniment = np.sin(2 * np.pi * 220 * t) * 0.5
    mixture = vocals + accompaniment

    # Separate
    separated, mask = separator.separate(mixture, vocals, mask_type='ibm')

    print("Baseline separator test completed")
    print(f"Input shape: {mixture.shape}")
    print(f"Output shape: {separated.shape}")
    print(f"Mask shape: {mask.shape}")