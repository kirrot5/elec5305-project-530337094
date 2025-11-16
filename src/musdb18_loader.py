"""
MUSDB18 Dataset Loader for Music Source Separation
Author: Kiro Chen (530337094)

Handles loading and preprocessing of MUSDB18 dataset
Dataset: https://sigsep.github.io/datasets/musdb.html
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class MUSDB18Loader:
    """
    Load and process MUSDB18 dataset for source separation

    MUSDB18 structure:
    musdb18/
        train/
            track1/
                mixture.wav
                vocals.wav
                bass.wav
                drums.wav
                other.wav
            track2/
            ...
        test/
            track1/
            ...
    """

    def __init__(self, root_path, subset='train', sr=22050, duration=None):
        """
        Args:
            root_path: Path to MUSDB18 dataset root
            subset: 'train' or 'test'
            sr: Target sample rate
            duration: Duration to load (seconds), None for full tracks
        """
        self.root_path = Path(root_path)
        self.subset = subset
        self.sr = sr
        self.duration = duration

        self.subset_path = self.root_path / subset

        if not self.subset_path.exists():
            print(f"Warning: Dataset path not found: {self.subset_path}")
            print("Please download MUSDB18 dataset from: https://sigsep.github.io/datasets/musdb.html")
            self.track_names = []
        else:
            # Get all track directories
            self.track_names = sorted([
                d.name for d in self.subset_path.iterdir()
                if d.is_dir()
            ])
            print(f"✓ Found {len(self.track_names)} tracks in {subset} set")

    def get_track_list(self) -> List[str]:
        """Get list of available track names"""
        return self.track_names

    def load_track(self, track_name: str, sources: Optional[List[str]] = None) -> Dict:
        """
        Load a single track with all sources

        Args:
            track_name: Name of track (e.g., 'A Classic Education - NightOwl')
            sources: List of sources to load.
                    Options: ['mixture', 'vocals', 'bass', 'drums', 'other']
                    If None, loads all sources

        Returns:
            track_data: Dictionary with audio arrays for each source
        """
        track_path = self.subset_path / track_name

        if not track_path.exists():
            raise ValueError(f"Track not found: {track_name}")

        if sources is None:
            sources = ['mixture', 'vocals', 'bass', 'drums', 'other']

        track_data = {
            'name': track_name,
            'sr': self.sr
        }

        for source in sources:
            source_file = track_path / f"{source}.wav"

            if not source_file.exists():
                print(f"Warning: {source}.wav not found for {track_name}")
                continue

            # Load audio
            audio, sr = librosa.load(
                str(source_file),
                sr=self.sr,
                mono=False,
                duration=self.duration
            )

            track_data[source] = audio

        return track_data

    def load_track_for_separation(self, track_name: str) -> Tuple[np.ndarray, Dict, int]:
        """
        Load track specifically for vocal/accompaniment separation task

        Args:
            track_name: Name of track

        Returns:
            mixture: Mixed audio
            sources: Dictionary with 'vocals' and 'accompaniment'
            sr: Sample rate
        """
        track_data = self.load_track(track_name,
                                     sources=['mixture', 'vocals', 'bass', 'drums', 'other'])

        mixture = track_data.get('mixture')
        vocals = track_data.get('vocals')

        # Combine bass, drums, other into accompaniment
        accompaniment = None
        for src in ['bass', 'drums', 'other']:
            if src in track_data:
                if accompaniment is None:
                    accompaniment = track_data[src].copy()
                else:
                    # Ensure same length
                    min_len = min(accompaniment.shape[-1], track_data[src].shape[-1])
                    accompaniment = accompaniment[..., :min_len] + track_data[src][..., :min_len]

        sources = {
            'vocals': vocals,
            'accompaniment': accompaniment
        }

        return mixture, sources, self.sr

    def create_dataset_splits(self, train_ratio=0.8, random_seed=42):
        """
        Split dataset into train/validation sets

        Args:
            train_ratio: Ratio of training data
            random_seed: Random seed for reproducibility

        Returns:
            train_tracks: List of training track names
            val_tracks: List of validation track names
        """
        np.random.seed(random_seed)

        tracks = self.track_names.copy()
        np.random.shuffle(tracks)

        n_train = int(len(tracks) * train_ratio)
        train_tracks = tracks[:n_train]
        val_tracks = tracks[n_train:]

        print(f"Dataset split:")
        print(f"  Training: {len(train_tracks)} tracks")
        print(f"  Validation: {len(val_tracks)} tracks")

        return train_tracks, val_tracks

    def extract_chunks(self, audio: np.ndarray, chunk_duration: float = 5.0,
                       overlap: float = 0.0) -> List[np.ndarray]:
        """
        Extract fixed-length chunks from audio for training

        Args:
            audio: Audio array
            chunk_duration: Chunk length in seconds
            overlap: Overlap between chunks in seconds

        Returns:
            chunks: List of audio chunks
        """
        chunk_samples = int(chunk_duration * self.sr)
        hop_samples = int((chunk_duration - overlap) * self.sr)

        chunks = []

        if audio.ndim == 1:
            # Mono
            for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
                chunk = audio[start:start + chunk_samples]
                if len(chunk) == chunk_samples:
                    chunks.append(chunk)
        else:
            # Stereo
            for start in range(0, audio.shape[1] - chunk_samples + 1, hop_samples):
                chunk = audio[:, start:start + chunk_samples]
                if chunk.shape[1] == chunk_samples:
                    chunks.append(chunk)

        return chunks

    def prepare_training_data(self, track_names: List[str],
                              chunk_duration: float = 5.0) -> Tuple[List, List]:
        """
        Prepare training data from multiple tracks

        Args:
            track_names: List of track names to process
            chunk_duration: Length of each training chunk

        Returns:
            mixtures: List of mixture chunks
            sources: List of source dictionaries with 'vocals' and 'accompaniment'
        """
        all_mixtures = []
        all_sources = []

        print(f"Preparing training data from {len(track_names)} tracks...")

        for i, track_name in enumerate(track_names):
            print(f"  Processing {i + 1}/{len(track_names)}: {track_name}")

            try:
                mixture, sources, sr = self.load_track_for_separation(track_name)

                # Extract chunks
                mix_chunks = self.extract_chunks(mixture, chunk_duration)
                voc_chunks = self.extract_chunks(sources['vocals'], chunk_duration)
                acc_chunks = self.extract_chunks(sources['accompaniment'], chunk_duration)

                # Ensure all have same number of chunks
                n_chunks = min(len(mix_chunks), len(voc_chunks), len(acc_chunks))

                for j in range(n_chunks):
                    all_mixtures.append(mix_chunks[j])
                    all_sources.append({
                        'vocals': voc_chunks[j],
                        'accompaniment': acc_chunks[j]
                    })

                print(f"    ✓ Extracted {n_chunks} chunks")

            except Exception as e:
                print(f"    ✗ Error processing {track_name}: {e}")
                continue

        print(f"\n✓ Total chunks prepared: {len(all_mixtures)}")

        return all_mixtures, all_sources


# Utility function to download MUSDB18 (placeholder)
def download_musdb18_instructions():
    """Print instructions for downloading MUSDB18"""
    instructions = """
    ========================================================================
    MUSDB18 Dataset Download Instructions
    ========================================================================

    1. Visit: https://sigsep.github.io/datasets/musdb.html

    2. Download MUSDB18-HQ (recommended) or MUSDB18

    3. Extract to a directory, e.g., './data/musdb18/'

    4. Directory structure should be:
       musdb18/
           train/
               A Classic Education - NightOwl/
                   mixture.wav
                   vocals.wav
                   bass.wav
                   drums.wav
                   other.wav
               ...
           test/
               ...

    5. Alternative: Use musdb Python package
       pip install musdb

       Then use:
       import musdb
       mus = musdb.DB(download=True)

    ========================================================================
    """
    print(instructions)


# Test the loader
if __name__ == "__main__":
    print("Testing MUSDB18 Loader...")

    # Print download instructions
    download_musdb18_instructions()

    # Test with dummy data (if dataset exists)
    dataset_path = "./data/musdb18"

    if os.path.exists(dataset_path):
        print(f"\nDataset found at: {dataset_path}")

        loader = MUSDB18Loader(dataset_path, subset='train', sr=22050)

        # Get track list
        tracks = loader.get_track_list()
        print(f"\nAvailable tracks: {len(tracks)}")
        if tracks:
            print(f"First 3 tracks: {tracks[:3]}")

            # Load first track
            print(f"\nLoading track: {tracks[0]}")
            mixture, sources, sr = loader.load_track_for_separation(tracks[0])

            print(f"  Mixture shape: {mixture.shape}")
            print(f"  Vocals shape: {sources['vocals'].shape}")
            print(f"  Accompaniment shape: {sources['accompaniment'].shape}")

            # Test chunking
            print("\nTesting chunk extraction...")
            chunks = loader.extract_chunks(mixture, chunk_duration=5.0)
            print(f"  Extracted {len(chunks)} chunks")
            if chunks:
                print(f"  Chunk shape: {chunks[0].shape}")
    else:
        print(f"\nDataset not found at: {dataset_path}")
        print("Use download_musdb18_instructions() for setup guide")