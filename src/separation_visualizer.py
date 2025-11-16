"""
Visualization Tools for Music Source Separation
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


class SeparationVisualizer:
    """Visualization tools for source separation"""

    def __init__(self, sr=22050, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length


    def plot_separation_results(self, mixture, vocals, accompaniment,
                                vocals_true=None, acc_true=None,
                                figsize=(16, 12)):
        """
        Plot comprehensive separation results:
        - waveforms
        - spectrograms
        - comparison plots
        - optional error analysis
        """
        # Handle stereo by taking first channel
        if mixture.ndim > 1:
            mixture = mixture[0] if mixture.shape[0] == 2 else mixture[:, 0]
        if vocals.ndim > 1:
            vocals = vocals[0] if vocals.shape[0] == 2 else vocals[:, 0]
        if accompaniment.ndim > 1:
            accompaniment = accompaniment[0] if accompaniment.shape[0] == 2 else accompaniment[:, 0]

        n_rows = 6 if vocals_true is not None else 4
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

        # Row 1: Mixture
        self._plot_waveform(axes[0, 0], mixture, "Mixture - Waveform")
        self._plot_spectrogram(axes[0, 1], mixture, "Mixture - Spectrogram")

        # Row 2: Separated vocals
        self._plot_waveform(axes[1, 0], vocals, "Separated Vocals - Waveform")
        self._plot_spectrogram(axes[1, 1], vocals, "Separated Vocals - Spectrogram")

        # Row 3: Separated accompaniment
        self._plot_waveform(axes[2, 0], accompaniment, "Separated Accompaniment - Waveform")
        self._plot_spectrogram(axes[2, 1], accompaniment, "Separated Accompaniment - Spectrogram")

        # Row 4: Comparison
        self._plot_comparison_waveform(axes[3, 0], mixture, vocals, accompaniment)
        self._plot_spectral_comparison(axes[3, 1], mixture, vocals, accompaniment)

        # Optional ground truth analysis
        if vocals_true is not None and acc_true is not None:
            if vocals_true.ndim > 1:
                vocals_true = vocals_true[0]
            if acc_true.ndim > 1:
                acc_true = acc_true[0]

            self._plot_waveform(axes[4, 0], vocals_true, "Ground Truth Vocals - Waveform", color='green')
            self._plot_spectrogram(axes[4, 1], vocals_true, "Ground Truth Vocals - Spectrogram")

            self._plot_error_analysis(axes[5, 0], vocals_true, vocals, "Vocals Error")
            self._plot_error_analysis(axes[5, 1], acc_true, accompaniment, "Accompaniment Error")

        plt.tight_layout()
        return fig


    def _plot_waveform(self, ax, audio, title, color='blue'):
        time = np.arange(len(audio)) / self.sr
        ax.plot(time, audio, alpha=0.7, linewidth=0.5, color=color)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.1, 1.1])

    def _plot_spectrogram(self, ax, audio, title):
        D = librosa.stft(audio, hop_length=self.hop_length)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            D_db, sr=self.sr, hop_length=self.hop_length,
            x_axis='time', y_axis='hz', ax=ax, cmap='viridis'
        )
        ax.set_title(title)
        ax.set_ylim([0, 8000])
        plt.colorbar(img, ax=ax, format='%+2.0f dB')

    def _plot_comparison_waveform(self, ax, mixture, vocals, accompaniment):
        min_len = min(len(mixture), len(vocals), len(accompaniment))
        mixture = mixture[:min_len]
        vocals = vocals[:min_len]
        accompaniment = accompaniment[:min_len]

        time = np.arange(min_len) / self.sr

        mixture_norm = mixture / (np.max(np.abs(mixture)) + 1e-10)
        vocals_norm = vocals / (np.max(np.abs(vocals)) + 1e-10)
        acc_norm = accompaniment / (np.max(np.abs(accompaniment)) + 1e-10)

        ax.plot(time, mixture_norm, alpha=0.3, label='Mixture', color='gray', linewidth=0.5)
        ax.plot(time, vocals_norm * 0.8, alpha=0.7, label='Vocals', color='red', linewidth=0.5)
        ax.plot(time, acc_norm * 0.8 - 1, alpha=0.7, label='Accompaniment', color='blue', linewidth=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title('Waveform Comparison')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_spectral_comparison(self, ax, mixture, vocals, accompaniment):
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sr)

        mix_spec = np.abs(np.fft.rfft(mixture[:n_fft], n=n_fft)) if len(mixture) >= n_fft else np.zeros_like(freqs)
        voc_spec = np.abs(np.fft.rfft(vocals[:n_fft], n=n_fft)) if len(vocals) >= n_fft else np.zeros_like(freqs)
        acc_spec = np.abs(np.fft.rfft(accompaniment[:n_fft], n=n_fft)) if len(accompaniment) >= n_fft else np.zeros_like(
            freqs)

        ax.plot(freqs, 20 * np.log10(mix_spec + 1e-10), alpha=0.5, label='Mixture', color='gray')
        ax.plot(freqs, 20 * np.log10(voc_spec + 1e-10), alpha=0.7, label='Vocals', color='red')
        ax.plot(freqs, 20 * np.log10(acc_spec + 1e-10), alpha=0.7, label='Accompaniment', color='blue')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title('Spectral Content Comparison')
        ax.set_xlim([0, 8000])
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_error_analysis(self, ax, true_signal, estimated_signal, title):
        min_len = min(len(true_signal), len(estimated_signal))
        true_signal = true_signal[:min_len]
        estimated_signal = estimated_signal[:min_len]

        error = true_signal - estimated_signal
        time = np.arange(min_len) / self.sr

        ax.plot(time, error, alpha=0.7, color='red', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        rms_error = np.sqrt(np.mean(error ** 2))
        ax.text(0.02, 0.98, f'RMS Error: {rms_error:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


    def plot_metrics_comparison(self, metrics_list, method_names, figsize=(12, 6)):
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        metric_names = ['SDR', 'SIR', 'SAR']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]

            vocals_values = [m[f'vocals_{metric_name}'] for m in metrics_list]
            acc_values = [m[f'accompaniment_{metric_name}'] for m in metrics_list]

            x = np.arange(len(method_names))
            width = 0.35

            ax.bar(x - width / 2, vocals_values, width, label='Vocals', color=colors[0], alpha=0.8)
            ax.bar(x + width / 2, acc_values, width, label='Accompaniment', color=colors[1], alpha=0.8)

            ax.set_xlabel('Method')
            ax.set_ylabel(f'{metric_name} (dB)')
            ax.set_title(f'{metric_name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

        plt.tight_layout()
        return fig


    def plot_audio_features(self, audio, title_prefix="Mixture",
                            frame_length=2048, hop_length=None):
        """
        Plot basic time-domain and frequency-domain audio features:

        - Short-time energy
        - Zero-crossing rate
        - Spectral centroid
        - Spectral flatness
        - Spectral rolloff
        """
        if hop_length is None:
            hop_length = self.hop_length

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Time-domain features
        frames = librosa.util.frame(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        )
        energy = np.sum(frames ** 2, axis=0)

        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Spectral features
        S = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
        centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr)[0]

        times = librosa.frames_to_time(np.arange(len(energy)), sr=self.sr, hop_length=hop_length)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(times, energy, label='Energy')
        axes[0].set_ylabel('Energy')
        axes[0].set_title(f'{title_prefix} - Short-time Energy')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, zcr, label='ZCR', color='orange')
        axes[1].set_ylabel('Zero-Crossing Rate')
        axes[1].set_title(f'{title_prefix} - Zero-Crossing Rate')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(times, centroid, label='Centroid')
        axes[2].set_ylabel('Hz')
        axes[2].set_title(f'{title_prefix} - Spectral Centroid')
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(times, flatness, label='Flatness', color='green')
        axes[3].plot(times, rolloff / np.max(rolloff + 1e-10), label='Rolloff (norm.)', color='red', alpha=0.7)
        axes[3].set_ylabel('Value')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title(f'{title_prefix} - Spectral Flatness & Rolloff (norm.)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


    def plot_modulation_spectrum(self, audio, title="Modulation Spectrum",
                                 n_fft=2048, hop_length=None,
                                 patch_size=(64, 64)):
        """
        Compute and plot a simple 2D modulation spectrum.

        Steps:
        1) Compute magnitude spectrogram |S(f, t)|
        2) Take log(1 + |S|)
        3) Optionally crop/resize to patch_size
        4) Apply 2D FFT and show |F(ω_f, ω_t)|

        This is a simplified version of the lecture's modulation spectrum /
        2D Welch idea.
        """
        if hop_length is None:
            hop_length = self.hop_length

        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_log = np.log1p(S)

        F, T = S_log.shape

        # Simple resizing to patch_size by cropping or pooling
        target_F, target_T = patch_size
        S_resized = S_log

        if F > target_F:
            S_resized = S_resized[:target_F, :]
        elif F < target_F:
            pad_F = target_F - F
            S_resized = np.pad(S_resized, ((0, pad_F), (0, 0)), mode='constant')

        if T > target_T:
            S_resized = S_resized[:, :target_T]
        elif T < target_T:
            pad_T = target_T - T
            S_resized = np.pad(S_resized, ((0, 0), (0, pad_T)), mode='constant')

        # 2D FFT
        MS = np.fft.fftshift(np.fft.fft2(S_resized))
        MS_mag = 20 * np.log10(np.abs(MS) + 1e-10)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        img = ax.imshow(MS_mag, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(title)
        ax.set_xlabel('Temporal Modulation bin')
        ax.set_ylabel('Spectral Modulation bin')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        return fig
