"""
Vocal Activity Detector - Baseline Methods
Author: Kiro Chen (530337094)
"""

import numpy as np
import librosa
from scipy.ndimage import median_filter

class VocalActivityDetector:
    """
    Baseline vocal activity detector using signal processing methods
    """

    def __init__(self, method='energy', threshold=0.5, sr=22050,
                 n_fft=2048, hop_length=512):
        """
        Args:
            method: Detection method ('energy', 'mfcc_variance', 'spectral')
            threshold: Detection threshold (0-1)
            sr: Sample rate
            n_fft: FFT size
            hop_length: Hop length
        """
        self.method = method
        self.threshold = threshold
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def detect(self, audio):
        """
        Main detection function

        Args:
            audio: Audio signal (1D array)

        Returns:
            predictions: Binary array (1=vocal, 0=instrumental)
            times: Time stamps for each frame
            confidence: Confidence scores
        """
        if self.method == 'energy':
            return self._detect_energy_based(audio)
        elif self.method == 'mfcc_variance':
            return self._detect_mfcc_based(audio)
        elif self.method == 'spectral':
            return self._detect_spectral_based(audio)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_energy_based(self, audio):
        """
        Energy-based detection
        Assumption: Vocal sections have higher energy
        """
        # Compute RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]

        # Normalize to [0, 1]
        rms_min, rms_max = np.min(rms), np.max(rms)
        confidence = (rms - rms_min) / (rms_max - rms_min + 1e-8)

        # Threshold
        predictions = (confidence > self.threshold).astype(int)

        # Get times
        times = librosa.frames_to_time(
            np.arange(len(predictions)),
            sr=self.sr,
            hop_length=self.hop_length
        )

        return predictions, times, confidence

    def _detect_mfcc_based(self, audio):
        """
        MFCC variance-based detection
        Assumption: Vocals have more dynamic MFCC patterns
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=13,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Compute variance across MFCC coefficients
        mfcc_var = np.var(mfcc, axis=0)

        # Normalize
        var_min, var_max = np.min(mfcc_var), np.max(mfcc_var)
        confidence = (mfcc_var - var_min) / (var_max - var_min + 1e-8)

        # Threshold
        predictions = (confidence > self.threshold).astype(int)

        times = librosa.frames_to_time(
            np.arange(len(predictions)),
            sr=self.sr,
            hop_length=self.hop_length
        )

        return predictions, times, confidence

    def _detect_spectral_based(self, audio):
        """
        Spectral-based detection
        Uses spectral centroid and bandwidth
        """
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]

        # Normalize both
        centroid_norm = (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid) + 1e-8)
        bandwidth_norm = (bandwidth - np.min(bandwidth)) / (np.max(bandwidth) - np.min(bandwidth) + 1e-8)

        # Combine (weighted average)
        confidence = 0.6 * centroid_norm + 0.4 * bandwidth_norm

        # Threshold
        predictions = (confidence > self.threshold).astype(int)

        times = librosa.frames_to_time(
            np.arange(len(predictions)),
            sr=self.sr,
            hop_length=self.hop_length
        )

        return predictions, times, confidence

    def post_process(self, predictions, min_duration_frames=10, median_size=5):
        """
        Post-process predictions to remove spurious detections

        Args:
            predictions: Binary predictions
            min_duration_frames: Minimum segment duration in frames
            median_size: Median filter size

        Returns:
            smoothed: Smoothed predictions
        """
        # Apply median filter to remove isolated frames
        smoothed = median_filter(predictions, size=median_size)

        # Remove short segments
        smoothed = self._remove_short_segments(smoothed, min_duration_frames)

        return smoothed.astype(int)

    def _remove_short_segments(self, predictions, min_frames):
        """Remove segments shorter than min_frames"""
        result = predictions.copy()

        # Find segment boundaries
        changes = np.diff(np.concatenate([[0], predictions, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        # Remove short segments
        for start, end in zip(starts, ends):
            if end - start < min_frames:
                result[start:end] = 0

        return result

    def get_segments(self, predictions, times):
        """
        Convert frame-level predictions to time segments

        Args:
            predictions: Binary predictions
            times: Time stamps

        Returns:
            segments: List of (label, start_time, end_time) tuples
        """
        segments = []
        in_vocal = False
        start_time = 0

        for i, (pred, time) in enumerate(zip(predictions, times)):
            if pred == 1 and not in_vocal:
                # Start of vocal segment
                start_time = time
                in_vocal = True
            elif pred == 0 and in_vocal:
                # End of vocal segment
                segments.append(('vocal', start_time, time))
                in_vocal = False

        # Handle last segment
        if in_vocal:
            segments.append(('vocal', start_time, times[-1]))

        # Add instrumental segments
        full_segments = []
        prev_end = 0

        for label, start, end in segments:
            if start > prev_end:
                full_segments.append(('instrumental', prev_end, start))
            full_segments.append((label, start, end))
            prev_end = end

        # Final instrumental segment
        if prev_end < times[-1]:
            full_segments.append(('instrumental', prev_end, times[-1]))

        return full_segments


# Test the detector
if __name__ == "__main__":
    print("Testing Vocal Activity Detector...")

    # Create test signal: alternating vocal and instrumental
    sr = 22050
    duration = 10
    t = np.linspace(0, duration, int(sr * duration))

    # Create alternating pattern
    audio = np.zeros_like(t)

    # Vocal sections (0-2s, 4-6s, 8-10s): rich harmonics
    for start in [0, 4, 8]:
        end = start + 2
        mask = (t >= start) & (t < end)
        audio[mask] = (np.sin(2 * np.pi * 440 * t[mask]) +
                       0.5 * np.sin(2 * np.pi * 880 * t[mask]) +
                       0.3 * np.sin(2 * np.pi * 1320 * t[mask]))

    # Instrumental sections (2-4s, 6-8s): simple bass
    for start in [2, 6]:
        end = start + 2
        mask = (t >= start) & (t < end)
        audio[mask] = 0.5 * np.sin(2 * np.pi * 220 * t[mask])

    # Test different methods
    methods = ['energy', 'mfcc_variance', 'spectral']

    for method in methods:
        print(f"\n--- Testing {method.upper()} method ---")

        detector = VocalActivityDetector(method=method, threshold=0.5, sr=sr)
        predictions, times, confidence = detector.detect(audio)

        # Post-process
        predictions_smooth = detector.post_process(predictions, min_duration_frames=5)

        # Get segments
        segments = detector.get_segments(predictions_smooth, times)

        print(f"  Detected {len(segments)} segments:")
        for label, start, end in segments[:5]:  # Show first 5
            print(f"    {label:15s}: {start:5.2f}s - {end:5.2f}s ({end-start:4.2f}s)")

        # Calculate rough accuracy
        vocal_detected = sum([end - start for label, start, end in segments if label == 'vocal'])
        print(f"  Total vocal time detected: {vocal_detected:.2f}s / 6.00s expected")