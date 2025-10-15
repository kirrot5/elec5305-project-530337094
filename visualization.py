"""
Visualization Tools for VAD
Author: Kiro Chen (530337094)
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

class VADVisualizer:
    """Visualization tools for vocal activity detection"""

    def __init__(self, sr=22050, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length

    def plot_detection_results(self, audio, predictions, times,
                               ground_truth=None, confidence=None,
                               figsize=(14, 8)):
        """
        Plot comprehensive detection results

        Args:
            audio: Audio signal
            predictions: Binary predictions
            times: Time stamps
            ground_truth: Optional ground truth labels
            confidence: Optional confidence scores
        """
        n_plots = 3 if ground_truth is None else 4
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

        # Plot 1: Waveform with predictions
        ax = axes[0]
        time_audio = np.arange(len(audio)) / self.sr
        ax.plot(time_audio, audio, alpha=0.5, color='gray', linewidth=0.5)
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform with VAD Results')
        ax.grid(True, alpha=0.3)

        # Overlay predictions
        for i in range(len(predictions)):
            if predictions[i] == 1:
                start_time = times[i]
                end_time = times[i+1] if i < len(times)-1 else times[i] + 0.023
                ax.axvspan(start_time, end_time, alpha=0.3, color='red', label='Vocal' if i == 0 else '')

        ax.legend(loc='upper right')

        # Plot 2: Spectrogram
        ax = axes[1]
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio, hop_length=self.hop_length)),
            ref=np.max
        )
        img = librosa.display.specshow(
            D, sr=self.sr, hop_length=self.hop_length,
            x_axis='time', y_axis='hz', ax=ax, cmap='viridis'
        )
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')
        ax.set_ylim([0, 8000])
        plt.colorbar(img, ax=ax, format='%+2.0f dB')

        # Overlay predictions on spectrogram
        for i in range(len(predictions)):
            if predictions[i] == 1:
                start_time = times[i]
                end_time = times[i+1] if i < len(times)-1 else times[i] + 0.023
                ax.axvspan(start_time, end_time, alpha=0.2, color='red')

        # Plot 3: Predictions timeline
        ax = axes[2]
        ax.plot(times, predictions, drawstyle='steps-post', linewidth=2, color='red', label='Predicted')

        if ground_truth is not None:
            ax.plot(times, ground_truth, drawstyle='steps-post', linewidth=2,
                   color='green', alpha=0.7, linestyle='--', label='Ground Truth')

        if confidence is not None:
            ax2 = ax.twinx()
            ax2.plot(times, confidence, alpha=0.5, color='blue', linewidth=1, label='Confidence')
            ax2.set_ylabel('Confidence', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set_ylim([0, 1])

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Prediction')
        ax.set_title('VAD Timeline')
        ax.set_ylim([-0.1, 1.1])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Plot 4: Ground truth comparison (if available)
        if ground_truth is not None:
            ax = axes[3]

            # Show agreement/disagreement
            agreement = (predictions == ground_truth).astype(int)
            colors = ['red' if a == 0 else 'green' for a in agreement]

            ax.scatter(times, predictions, c=colors, alpha=0.6, s=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Prediction')
            ax.set_title('Prediction vs Ground Truth (Green=Correct, Red=Error)')
            ax.set_ylim([-0.1, 1.1])
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_segments(self, segments, total_duration, figsize=(12, 4)):
        """
        Plot segment timeline

        Args:
            segments: List of (label, start, end) tuples
            total_duration: Total audio duration
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = {'vocal': 'red', 'instrumental': 'blue'}

        for label, start, end in segments:
            ax.barh(0, end - start, left=start, height=0.5,
                   color=colors.get(label, 'gray'),
                   alpha=0.7, edgecolor='black', linewidth=1)

            # Add label in the middle
            mid = (start + end) / 2
            ax.text(mid, 0, label[:4], ha='center', va='center',
                   fontsize=8, fontweight='bold')

        ax.set_xlim([0, total_duration])
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlabel('Time (s)')
        ax.set_yticks([])
        ax.set_title('Segment Timeline')
        ax.grid(True, axis='x', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Vocal'),
            Patch(facecolor='blue', alpha=0.7, label='Instrumental')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return fig

    def plot_feature_importance(self, feature_names, importances, top_n=20, figsize=(10, 8)):
        """
        Plot feature importance (for ML models)

        Args:
            feature_names: List of feature names
            importances: Feature importance values
            top_n: Number of top features to show
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(range(top_n), importances[indices], align='center')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, cm, classes=['Instrumental', 'Vocal'], figsize=(8, 6)):
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix (2x2 numpy array)
            classes: Class names
        """
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        # Set ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted Label',
               ylabel='True Label',
               title='Confusion Matrix')

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=20, fontweight='bold')

        plt.tight_layout()
        return fig


# Test visualizer
if __name__ == "__main__":
    print("Testing VAD Visualizer...")

    # Create test audio
    sr = 22050
    duration = 5
    t = np.linspace(0, duration, int(sr * duration))

    # Alternating vocal/instrumental pattern
    audio = np.zeros_like(t)
    audio[int(0*sr):int(2*sr)] = np.sin(2 * np.pi * 440 * t[int(0*sr):int(2*sr)])  # Vocal
    audio[int(3*sr):int(5*sr)] = np.sin(2 * np.pi * 220 * t[int(3*sr):int(5*sr)])  # Inst

    # Create predictions
    n_frames = 200
    times = np.linspace(0, duration, n_frames)
    predictions = np.zeros(n_frames)
    predictions[0:80] = 1    # Vocal
    predictions[120:200] = 1  # Vocal

    ground_truth = np.zeros(n_frames)
    ground_truth[0:90] = 1   # Slightly different
    ground_truth[130:200] = 1

    confidence = np.random.rand(n_frames) * 0.3 + 0.5
    confidence[predictions == 1] += 0.2

    # Visualize
    visualizer = VADVisualizer(sr=sr)

    print("\nGenerating detection results plot...")
    fig1 = visualizer.plot_detection_results(
        audio, predictions, times,
        ground_truth=ground_truth,
        confidence=confidence
    )

    print("Generating segment timeline...")
    segments = [
        ('vocal', 0.0, 2.0),
        ('instrumental', 2.0, 3.0),
        ('vocal', 3.0, 5.0)
    ]
    fig2 = visualizer.plot_segments(segments, duration)

    print("Generating confusion matrix...")
    cm = np.array([[150, 30], [20, 100]])
    fig3 = visualizer.plot_confusion_matrix(cm)

    print("✓ All visualizations generated successfully!")
    print("  (Close the plot windows to continue)")

    plt.show()