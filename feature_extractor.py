"""
Audio Feature Extraction for Vocal Activity Detection
Author: Kiro Chen (530337094)
"""

import numpy as np
import librosa

class AudioFeatureExtractor:
    """Extract audio features for VAD task"""

    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13):
        """
        Args:
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Hop length
            n_mfcc: Number of MFCC coefficients
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def extract_all_features(self, audio):
        """
        Extract comprehensive feature set for VAD

        Args:
            audio: Audio signal (1D numpy array)

        Returns:
            features: Dictionary of features
            times: Time stamps for each frame
        """
        # 1. MFCC - Mel Frequency Cepstral Coefficients
        # Good for capturing vocal timbre
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # 2. MFCC Delta (velocity)
        mfcc_delta = librosa.feature.delta(mfcc)

        # 3. MFCC Delta-Delta (acceleration)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # 4. Spectral Centroid
        # Vocals typically have higher spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # 5. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # 6. Zero Crossing Rate
        # Vocals have more zero crossings than bass instruments
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )

        # 7. RMS Energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )

        # 8. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # 9. Spectral Contrast
        # Measures peaks vs valleys in spectrum
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # 10. Chroma features
        # Useful for pitch-based detection
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Calculate time stamps
        n_frames = mfcc.shape[1]
        times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Combine all features
        features = {
            'mfcc': mfcc,  # (13, n_frames)
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zcr': zcr,
            'rms': rms,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'chroma': chroma,
            'times': times
        }

        return features

    def extract_frame_features(self, audio, aggregate=True):
        """
        Extract features and optionally aggregate to per-frame feature vector

        Args:
            audio: Audio signal
            aggregate: If True, return (n_features, n_frames) array

        Returns:
            feature_matrix: Aggregated features or dict
        """
        features = self.extract_all_features(audio)

        if not aggregate:
            return features

        # Stack all features into one matrix
        feature_list = []

        # MFCCs and deltas
        feature_list.append(features['mfcc'])
        feature_list.append(features['mfcc_delta'])
        feature_list.append(features['mfcc_delta2'])

        # Other features
        feature_list.append(features['spectral_centroid'])
        feature_list.append(features['spectral_rolloff'])
        feature_list.append(features['zcr'])
        feature_list.append(features['rms'])
        feature_list.append(features['spectral_bandwidth'])
        feature_list.append(features['spectral_contrast'])
        feature_list.append(features['chroma'])

        # Stack: (total_features, n_frames)
        feature_matrix = np.vstack(feature_list)

        return feature_matrix, features['times']

    def get_feature_names(self):
        """Return list of feature names"""
        names = []

        # MFCC
        for i in range(self.n_mfcc):
            names.append(f'mfcc_{i}')

        # MFCC deltas
        for i in range(self.n_mfcc):
            names.append(f'mfcc_delta_{i}')

        # MFCC delta-deltas
        for i in range(self.n_mfcc):
            names.append(f'mfcc_delta2_{i}')

        # Other features
        names.extend([
            'spectral_centroid',
            'spectral_rolloff',
            'zcr',
            'rms',
            'spectral_bandwidth'
        ])

        # Spectral contrast (7 bands)
        for i in range(7):
            names.append(f'spectral_contrast_{i}')

        # Chroma (12 pitch classes)
        for i in range(12):
            names.append(f'chroma_{i}')

        return names


# Test the feature extractor
if __name__ == "__main__":
    print("Testing Audio Feature Extractor...")

    # Create dummy audio
    sr = 22050
    duration = 5
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate vocal: harmonics at 220, 440, 660 Hz
    audio = (np.sin(2 * np.pi * 220 * t) +
             0.5 * np.sin(2 * np.pi * 440 * t) +
             0.3 * np.sin(2 * np.pi * 660 * t))

    # Add some noise
    audio += 0.05 * np.random.randn(len(audio))

    # Extract features
    extractor = AudioFeatureExtractor(sr=sr)
    features = extractor.extract_all_features(audio)

    print(f"\n✓ Feature extraction successful!")
    print(f"  - MFCC shape: {features['mfcc'].shape}")
    print(f"  - Number of frames: {len(features['times'])}")
    print(f"  - Time range: {features['times'][0]:.2f}s to {features['times'][-1]:.2f}s")

    # Test aggregated features
    feature_matrix, times = extractor.extract_frame_features(audio, aggregate=True)
    print(f"\n✓ Aggregated feature matrix: {feature_matrix.shape}")
    print(f"  - Total features per frame: {feature_matrix.shape[0]}")

    feature_names = extractor.get_feature_names()
    print(f"  - Feature names (first 5): {feature_names[:5]}")
    print(f"  - Total number of features: {len(feature_names)}")