"""
Audio Loading and Preprocessing for Music Source Separation

"""

import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path


class AudioLoader:
    """Load and preprocess audio files for source separation"""

    def __init__(self, sr=22050, mono=False):
        """
        Args:
            sr: Target sample rate
            mono: If True, convert to mono; if False, keep stereo
        """
        self.sr = sr
        self.mono = mono

    def load_audio(self, filepath, duration=None, offset=0.0):
        """
        Load audio file (supports WAV, MP3, M4A, FLAC, OGG, etc.)

        Args:
            filepath: Path to audio file (any format supported by librosa)
            duration: Duration to load (seconds), None for full file
            offset: Start time (seconds)

        Returns:
            audio: Audio signal (channels, samples) or (samples,) if mono
            sr: Sample rate
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        # Get file extension
        file_ext = os.path.splitext(filepath)[1].lower()

        print(f"Loading {file_ext} file...")

        # Load audio (librosa supports MP3, WAV, FLAC, OGG, M4A, etc.)
        audio, sr = librosa.load(
            filepath,
            sr=self.sr,
            mono=self.mono,
            duration=duration,
            offset=offset
        )

        print(f"✓ Loaded: {Path(filepath).name}")
        print(f"  Format: {file_ext}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Shape: {audio.shape}")
        print(f"  Duration: {len(audio) / sr:.2f}s")

        return audio, sr

    def load_mixture(self, filepath, **kwargs):
        """Load mixture audio (alias for load_audio)"""
        return self.load_audio(filepath, **kwargs)

    def load_sources(self, vocals_path, accompaniment_path, **kwargs):
        """
        Load separated source files (for evaluation)

        Args:
            vocals_path: Path to vocals WAV file
            accompaniment_path: Path to accompaniment WAV file

        Returns:
            vocals: Vocal audio
            accompaniment: Accompaniment audio
            sr: Sample rate
        """
        vocals, sr = self.load_audio(vocals_path, **kwargs)
        accompaniment, _ = self.load_audio(accompaniment_path, **kwargs)

        return vocals, accompaniment, sr

    def save_audio(self, audio, filepath, sr=None):
        """
        Save audio to WAV file

        Args:
            audio: Audio signal
            filepath: Output path
            sr: Sample rate (uses self.sr if None)
        """
        if sr is None:
            sr = self.sr

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # Ensure correct shape for soundfile
        if audio.ndim == 1:
            # Mono: (samples,)
            audio_to_save = audio
        elif audio.shape[0] == 2:
            # Stereo: (2, samples) -> transpose to (samples, 2)
            audio_to_save = audio.T
        else:
            audio_to_save = audio

        # Save
        sf.write(filepath, audio_to_save, sr)
        print(f"✓ Saved: {filepath}")

    def normalize_audio(self, audio, target_db=-20):
        """
        Normalize audio to target dB level

        Args:
            audio: Audio signal
            target_db: Target level in dB

        Returns:
            normalized: Normalized audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))

        if rms == 0:
            return audio

        # Convert target dB to linear scale
        target_linear = 10 ** (target_db / 20)

        # Scale audio
        normalized = audio * (target_linear / rms)

        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val * 0.95

        return normalized

    def split_audio_chunks(self, audio, chunk_duration=30.0, overlap=5.0):
        """
        Split long audio into overlapping chunks for processing

        Args:
            audio: Audio signal
            chunk_duration: Chunk length in seconds
            overlap: Overlap duration in seconds

        Returns:
            chunks: List of audio chunks
            positions: List of (start_sample, end_sample) tuples
        """
        chunk_samples = int(chunk_duration * self.sr)
        overlap_samples = int(overlap * self.sr)
        hop_samples = chunk_samples - overlap_samples

        chunks = []
        positions = []

        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))

            if audio.ndim == 1:
                chunk = audio[start:end]
            else:
                chunk = audio[:, start:end]

            chunks.append(chunk)
            positions.append((start, end))

            start += hop_samples

            # Break if we've covered the whole audio
            if end >= len(audio):
                break

        print(f"✓ Split into {len(chunks)} chunks")
        return chunks, positions

    def merge_chunks(self, chunks, positions, total_length, overlap_samples=None):
        """
        Merge overlapping chunks using crossfade

        Args:
            chunks: List of audio chunks
            positions: List of (start, end) positions
            total_length: Total output length
            overlap_samples: Overlap length for crossfading

        Returns:
            merged: Merged audio
        """
        if overlap_samples is None:
            overlap_samples = int(5.0 * self.sr)

        # Determine shape
        if chunks[0].ndim == 1:
            merged = np.zeros(total_length)
        else:
            merged = np.zeros((chunks[0].shape[0], total_length))

        weights = np.zeros(total_length)

        for chunk, (start, end) in zip(chunks, positions):
            chunk_len = end - start

            # Create window for smooth blending
            window = np.ones(chunk_len)

            # Fade in at start (except first chunk)
            if start > 0:
                fade_len = min(overlap_samples, chunk_len // 2)
                window[:fade_len] = np.linspace(0, 1, fade_len)

            # Fade out at end (except last chunk)
            if end < total_length:
                fade_len = min(overlap_samples, chunk_len // 2)
                window[-fade_len:] = np.linspace(1, 0, fade_len)

            # Add weighted chunk
            if chunk.ndim == 1:
                merged[start:end] += chunk * window
                weights[start:end] += window
            else:
                merged[:, start:end] += chunk * window[np.newaxis, :]
                weights[start:end] += window

        # Normalize by weights
        weights[weights == 0] = 1  # Avoid division by zero
        if merged.ndim == 1:
            merged = merged / weights
        else:
            merged = merged / weights[np.newaxis, :]

        return merged


# Test the loader
if __name__ == "__main__":
    print("Testing Audio Loader...")

    # Create test WAV file
    print("\n1. Creating test WAV file...")
    sr = 22050
    duration = 5
    t = np.linspace(0, duration, int(sr * duration))

    # Create stereo test signal
    left = np.sin(2 * np.pi * 440 * t)  # A4 in left channel
    right = np.sin(2 * np.pi * 554.37 * t)  # C#5 in right channel
    stereo = np.vstack([left, right])

    # Save test file
    test_path = "test_audio.wav"
    loader = AudioLoader(sr=sr, mono=False)
    loader.save_audio(stereo, test_path, sr=sr)

    # Load it back
    print("\n2. Loading test file...")
    audio, sr_loaded = loader.load_audio(test_path)

    # Test normalization
    print("\n3. Testing normalization...")
    normalized = loader.normalize_audio(audio)
    print(f"  Original RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"  Normalized RMS: {np.sqrt(np.mean(normalized**2)):.4f}")

    # Test chunking
    print("\n4. Testing chunking...")
    chunks, positions = loader.split_audio_chunks(audio, chunk_duration=2.0, overlap=0.5)
    print(f"  Chunk 0 shape: {chunks[0].shape}")

    # Test merging
    print("\n5. Testing merging...")
    merged = loader.merge_chunks(chunks, positions, audio.shape[-1])
    print(f"  Merged shape: {merged.shape}")
    print(f"  Reconstruction error: {np.mean((audio - merged)**2):.6f}")

    # Clean up
    os.remove(test_path)
    print("\n✓ All tests passed!")