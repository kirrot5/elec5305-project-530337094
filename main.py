
import os
import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from src.nmf_separator import NMFSeparator, hpss_separate, compare_methods
from src.separation_visualizer import SeparationVisualizer

# === Default input file for your project ===
DEFAULT_FILE = r"C:\Users\17890\Desktop\5305\data\song.mp3"


def separate_audio(input_file, output_dir='outputs', method='nmf', analyze=False):
    """
    Separate a single audio file.

    Args:
        input_file: Path to input audio file
        output_dir: Output directory
        method: 'nmf', 'demucs', 'hpss'
        analyze: if True, generate additional visualizations (features & modulation spectrum)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📂 Loading: {input_file}")
    audio, sr = librosa.load(input_file, sr=22050, mono=True)
    print(f"   Duration: {len(audio) / sr:.2f} s")
    print(f"   Sample rate: {sr} Hz")

    # Choose separation method
    if method in ['nmf', 'demucs']:
        use_demucs = (method == 'demucs')
        separator = NMFSeparator(
            n_components_vocals=30,
            n_components_acc=50,
            sr=sr,
            use_demucs=use_demucs
        )
        print(f"\n🎵 Separating using {method.upper()}...")
        vocals, accompaniment = separator.separate(audio, sr, method=method)
    elif method == 'hpss':
        print("\n🎵 Separating using HPSS (harmonic/percussive)...")
        harm, perc = hpss_separate(audio, sr)
        # Approximate naming: treat harmonic as "vocals-like", percussive as "accompaniment-like"
        vocals, accompaniment = harm, perc
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save results
    input_name = Path(input_file).stem
    vocals_path = f'{output_dir}/{input_name}_vocals_{method}.wav'
    acc_path = f'{output_dir}/{input_name}_accompaniment_{method}.wav'

    sf.write(vocals_path, vocals, sr)
    sf.write(acc_path, accompaniment, sr)

    print(f"\n✅ Results saved:")
    print(f"   Vocals-like:   {vocals_path}")
    print(f"   Accompaniment: {acc_path}")

    # Quick spectrograms (Mixture / Vocals / Accompaniment)
    visualize_spectrograms(audio, vocals, accompaniment, sr, output_dir, method)

    # Optional deeper analysis: audio features + modulation spectrum
    if analyze:
        visualizer = SeparationVisualizer(sr=sr)
        print("   Generating detailed analysis plots (features + modulation spectrum)...")

        fig_sep = visualizer.plot_separation_results(audio, vocals, accompaniment)
        fig_sep.savefig(f'{output_dir}/{input_name}_{method}_separation_overview.png',
                        dpi=150, bbox_inches='tight')
        plt.close(fig_sep)

        fig_feat_mix = visualizer.plot_audio_features(audio, title_prefix="Mixture")
        fig_feat_mix.savefig(f'{output_dir}/{input_name}_{method}_features_mixture.png',
                             dpi=150, bbox_inches='tight')
        plt.close(fig_feat_mix)

        fig_feat_voc = visualizer.plot_audio_features(vocals, title_prefix="Vocals-like")
        fig_feat_voc.savefig(f'{output_dir}/{input_name}_{method}_features_vocals.png',
                             dpi=150, bbox_inches='tight')
        plt.close(fig_feat_voc)

        fig_ms_mix = visualizer.plot_modulation_spectrum(audio, title="Mixture Modulation Spectrum")
        fig_ms_mix.savefig(f'{output_dir}/{input_name}_{method}_modspec_mixture.png',
                           dpi=150, bbox_inches='tight')
        plt.close(fig_ms_mix)

        print("   Analysis figures saved.")

    return vocals, accompaniment


def visualize_spectrograms(mixture, vocals, acc, sr, output_dir, method):
    """
    Visualize spectrograms of mixture and separated sources.
    This is a simple 3-row plot used in the assignment.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Mixture
    D_mix = librosa.amplitude_to_db(
        np.abs(librosa.stft(mixture)),
        ref=np.max
    )
    img1 = librosa.display.specshow(
        D_mix,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=axes[0]
    )
    axes[0].set_title('Mixture', fontsize=14)
    plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # Vocals
    D_vocals = librosa.amplitude_to_db(
        np.abs(librosa.stft(vocals)),
        ref=np.max
    )
    img2 = librosa.display.specshow(
        D_vocals,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=axes[1]
    )
    axes[1].set_title('Vocals-like', fontsize=14)
    plt.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    # Accompaniment
    D_acc = librosa.amplitude_to_db(
        np.abs(librosa.stft(acc)),
        ref=np.max
    )
    img3 = librosa.display.specshow(
        D_acc,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=axes[2]
    )
    axes[2].set_title('Accompaniment-like', fontsize=14)
    plt.colorbar(img3, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()

    plot_path = f'{output_dir}/spectrogram_{method}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Spectrogram:   {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Music Source Separation using NMF / HPSS / (optional) Demucs'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=DEFAULT_FILE,
        help=f'Input audio file (default: {DEFAULT_FILE})'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['nmf', 'demucs', 'hpss', 'both'],
        default='nmf',   # 推荐默认用 NMF，避免 Demucs 环境问题
        help='Separation method (default: nmf)'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare NMF/HPSS/(Demucs) side-by-side (runtime only)'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Generate additional analysis plots (features + modulation spectrum)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🎵 Music Source Separation Project")
    print("=" * 60)

    if args.compare:
        compare_methods(args.input, args.output)

    if args.method == 'both':
        print("\n🔹 Running NMF...")
        separate_audio(args.input, args.output, method='nmf', analyze=args.analyze)

        print("\n🔹 Running HPSS...")
        separate_audio(args.input, args.output, method='hpss', analyze=args.analyze)
    else:
        separate_audio(args.input, args.output, method=args.method, analyze=args.analyze)

    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
