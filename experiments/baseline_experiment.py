"""
Baseline Experiments for Music Source Separation
Author: Kiro Chen (530337094)

This script runs baseline experiments using NMF-based separation methods.
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_loader import AudioLoader
from src.nmf_separator import NMFSeparator
from src.separation_evaluator import compute_sdr, compute_sir, compute_sar


def visualize_separation_results(mixture, vocals, acc, sr, output_path):
    """
    Visualize separation results with spectrograms

    Args:
        mixture: Mixed audio signal
        vocals: Separated vocals
        acc: Separated accompaniment
        sr: Sample rate
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Mixture spectrogram
    D_mix = librosa.amplitude_to_db(
        np.abs(librosa.stft(mixture)),
        ref=np.max
    )
    img1 = librosa.display.specshow(
        D_mix,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=axes[0],
        cmap='viridis'
    )
    axes[0].set_title('Mixture Spectrogram', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # Vocals spectrogram
    D_vocals = librosa.amplitude_to_db(
        np.abs(librosa.stft(vocals)),
        ref=np.max
    )
    img2 = librosa.display.specshow(
        D_vocals,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=axes[1],
        cmap='viridis'
    )
    axes[1].set_title('Separated Vocals', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    # Accompaniment spectrogram
    D_acc = librosa.amplitude_to_db(
        np.abs(librosa.stft(acc)),
        ref=np.max
    )
    img3 = librosa.display.specshow(
        D_acc,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=axes[2],
        cmap='viridis'
    )
    axes[2].set_title('Separated Accompaniment', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(img3, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Spectrogram saved to: {output_path}")
    plt.close()


def compute_separation_metrics(vocals_true, acc_true, vocals_pred, acc_pred):
    """
    Compute comprehensive separation metrics

    Args:
        vocals_true: Ground truth vocals
        acc_true: Ground truth accompaniment
        vocals_pred: Predicted vocals
        acc_pred: Predicted accompaniment

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    print("\n📊 Computing separation metrics...")

    # SDR (Signal-to-Distortion Ratio)
    try:
        sdr_vocals = compute_sdr(vocals_true, vocals_pred)
        sdr_acc = compute_sdr(acc_true, acc_pred)
        metrics['sdr_vocals'] = sdr_vocals
        metrics['sdr_acc'] = sdr_acc
        print(f"   SDR Vocals:        {sdr_vocals:.2f} dB")
        print(f"   SDR Accompaniment: {sdr_acc:.2f} dB")
    except Exception as e:
        print(f"   ⚠️ Failed to compute SDR: {e}")

    # SIR (Signal-to-Interference Ratio)
    try:
        sir_vocals = compute_sir(vocals_true, vocals_pred, acc_true)
        sir_acc = compute_sir(acc_true, acc_pred, vocals_true)
        metrics['sir_vocals'] = sir_vocals
        metrics['sir_acc'] = sir_acc
        print(f"   SIR Vocals:        {sir_vocals:.2f} dB")
        print(f"   SIR Accompaniment: {sir_acc:.2f} dB")
    except Exception as e:
        print(f"   ⚠️ Failed to compute SIR: {e}")

    # SAR (Signal-to-Artifacts Ratio)
    try:
        sar_vocals = compute_sar(vocals_true, vocals_pred)
        sar_acc = compute_sar(acc_true, acc_pred)
        metrics['sar_vocals'] = sar_vocals
        metrics['sar_acc'] = sar_acc
        print(f"   SAR Vocals:        {sar_vocals:.2f} dB")
        print(f"   SAR Accompaniment: {sar_acc:.2f} dB")
    except Exception as e:
        print(f"   ⚠️ Failed to compute SAR: {e}")

    return metrics


def test_basic_nmf_separation(input_file, output_dir='outputs/test_basic_nmf'):
    """
    Test 1: Basic NMF Separation

    Args:
        input_file: Input audio file path
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("TEST 1: Basic NMF Separation")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    print(f"\n📂 Loading: {input_file}")
    loader = AudioLoader()
    mixture, sr = loader.load_audio(input_file)
    print(f"   Duration: {len(mixture)/sr:.2f}s, Sample rate: {sr}Hz")

    # Initialize NMF separator
    separator = NMFSeparator(
        n_components_vocals=30,
        n_components_acc=50,
        sr=sr,
        use_demucs=False
    )

    # Separate
    print("\n🎵 Separating sources...")
    start_time = time.time()
    vocals, acc = separator.separate(mixture, sr)
    elapsed_time = time.time() - start_time

    print(f"✅ Separation completed in {elapsed_time:.2f}s")

    # Save results
    input_name = Path(input_file).stem
    vocals_path = f'{output_dir}/{input_name}_vocals.wav'
    acc_path = f'{output_dir}/{input_name}_acc.wav'

    sf.write(vocals_path, vocals, sr)
    sf.write(acc_path, acc, sr)

    print(f"\n💾 Results saved:")
    print(f"   Vocals:        {vocals_path}")
    print(f"   Accompaniment: {acc_path}")

    # Visualize
    spec_path = f'{output_dir}/{input_name}_spectrogram.png'
    visualize_separation_results(mixture, vocals, acc, sr, spec_path)

    return {
        'method': 'basic_nmf',
        'time': elapsed_time,
        'vocals': vocals,
        'acc': acc
    }


def test_wiener_filtering(input_file, output_dir='outputs/test_wiener'):
    """
    Test 2: NMF with Wiener Filtering Enhancement

    Args:
        input_file: Input audio file path
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("TEST 2: NMF with Wiener Filtering")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    print(f"\n📂 Loading: {input_file}")
    loader = AudioLoader()
    mixture, sr = loader.load_audio(input_file)

    # Initialize separator
    separator = NMFSeparator(
        n_components_vocals=30,
        n_components_acc=50,
        sr=sr,
        use_demucs=False
    )

    # Separate with Wiener filtering
    print("\n🎵 Separating with Wiener filtering...")
    start_time = time.time()
    vocals, acc = separator.separate_wiener(mixture, sr)
    elapsed_time = time.time() - start_time

    print(f"✅ Separation completed in {elapsed_time:.2f}s")

    # Save results
    input_name = Path(input_file).stem
    vocals_path = f'{output_dir}/{input_name}_vocals_wiener.wav'
    acc_path = f'{output_dir}/{input_name}_acc_wiener.wav'

    sf.write(vocals_path, vocals, sr)
    sf.write(acc_path, acc, sr)

    print(f"\n💾 Results saved:")
    print(f"   Vocals:        {vocals_path}")
    print(f"   Accompaniment: {acc_path}")

    # Visualize
    spec_path = f'{output_dir}/{input_name}_spectrogram_wiener.png'
    visualize_separation_results(mixture, vocals, acc, sr, spec_path)

    return {
        'method': 'wiener',
        'time': elapsed_time,
        'vocals': vocals,
        'acc': acc
    }


def test_different_nmf_components(input_file, output_dir='outputs/test_components'):
    """
    Test 3: Compare Different NMF Component Numbers

    Args:
        input_file: Input audio file path
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("TEST 3: Different NMF Component Configurations")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    print(f"\n📂 Loading: {input_file}")
    loader = AudioLoader()
    mixture, sr = loader.load_audio(input_file)

    # Test different component configurations
    configs = [
        {'n_components_vocals': 20, 'n_components_acc': 30, 'name': 'small'},
        {'n_components_vocals': 30, 'n_components_acc': 50, 'name': 'medium'},
        {'n_components_vocals': 50, 'n_components_acc': 80, 'name': 'large'},
    ]

    results = []

    for config in configs:
        print(f"\n🔧 Testing configuration: {config['name']}")
        print(f"   Vocals components: {config['n_components_vocals']}")
        print(f"   Acc components:    {config['n_components_acc']}")

        # Initialize separator
        separator = NMFSeparator(
            n_components_vocals=config['n_components_vocals'],
            n_components_acc=config['n_components_acc'],
            sr=sr,
            use_demucs=False
        )

        # Separate
        start_time = time.time()
        vocals, acc = separator.separate(mixture, sr)
        elapsed_time = time.time() - start_time

        print(f"✅ Completed in {elapsed_time:.2f}s")

        # Save results
        input_name = Path(input_file).stem
        vocals_path = f'{output_dir}/{input_name}_vocals_{config["name"]}.wav'
        acc_path = f'{output_dir}/{input_name}_acc_{config["name"]}.wav'

        sf.write(vocals_path, vocals, sr)
        sf.write(acc_path, acc, sr)

        results.append({
            'config': config['name'],
            'time': elapsed_time,
            'n_vocals': config['n_components_vocals'],
            'n_acc': config['n_components_acc']
        })

    # Print summary
    print("\n" + "-"*80)
    print("Configuration Comparison Summary:")
    print("-"*80)
    for result in results:
        print(f"{result['config']:10s} | "
              f"Vocals: {result['n_vocals']:2d} | "
              f"Acc: {result['n_acc']:2d} | "
              f"Time: {result['time']:.2f}s")

    return results


def run_baseline_experiments_synthetic(vocals_file, acc_file, output_dir='outputs/baseline_synthetic'):
    """
    Run baseline experiments with synthetic data (ground truth available)

    Args:
        vocals_file: Path to clean vocals audio
        acc_file: Path to clean accompaniment audio
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("BASELINE EXPERIMENTS - Synthetic Data (with Ground Truth)")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth sources
    print(f"\n📂 Loading ground truth sources...")
    loader = AudioLoader()
    vocals_true, sr = loader.load_audio(vocals_file)
    acc_true, sr = loader.load_audio(acc_file)

    print(f"   Vocals duration: {len(vocals_true)/sr:.2f}s")
    print(f"   Acc duration:    {len(acc_true)/sr:.2f}s")

    # Create mixture
    min_length = min(len(vocals_true), len(acc_true))
    vocals_true = vocals_true[:min_length]
    acc_true = acc_true[:min_length]
    mixture = vocals_true + acc_true

    print(f"\n🎵 Created synthetic mixture: {min_length/sr:.2f}s")

    # Save mixture
    mixture_path = f'{output_dir}/mixture.wav'
    sf.write(mixture_path, mixture, sr)
    print(f"   Mixture saved to: {mixture_path}")

    # Initialize separator
    separator = NMFSeparator(
        n_components_vocals=30,
        n_components_acc=50,
        sr=sr,
        use_demucs=False
    )

    # Separate
    print("\n🎵 Separating sources...")
    vocals_pred, acc_pred = separator.separate(mixture, sr)

    # Save predictions
    sf.write(f'{output_dir}/vocals_predicted.wav', vocals_pred, sr)
    sf.write(f'{output_dir}/acc_predicted.wav', acc_pred, sr)

    # Compute metrics
    metrics = compute_separation_metrics(
        vocals_true, acc_true,
        vocals_pred, acc_pred
    )

    # Visualize
    spec_path = f'{output_dir}/spectrogram_comparison.png'
    visualize_separation_results(mixture, vocals_pred, acc_pred, sr, spec_path)

    return metrics


def run_baseline_experiments_real(input_file, output_dir='outputs/baseline_real'):
    """
    Run baseline experiments on real audio (no ground truth)

    Args:
        input_file: Input audio file path
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("BASELINE EXPERIMENTS - Real Audio")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    print(f"\n📂 Loading: {input_file}")
    loader = AudioLoader()
    mixture, sr = loader.load_audio(input_file)

    print(f"   Duration: {len(mixture)/sr:.2f}s")
    print(f"   Sample rate: {sr}Hz")

    # Initialize separator
    separator = NMFSeparator(
        n_components_vocals=30,
        n_components_acc=50,
        sr=sr,
        use_demucs=False
    )

    # Test 1: Basic NMF
    print("\n" + "-"*80)
    print("Running: Basic NMF Separation")
    print("-"*80)

    start_time = time.time()
    vocals_nmf, acc_nmf = separator.separate(mixture, sr)
    time_nmf = time.time() - start_time

    print(f"✅ Completed in {time_nmf:.2f}s")

    # Save NMF results
    input_name = Path(input_file).stem
    nmf_dir = f'{output_dir}/nmf'
    os.makedirs(nmf_dir, exist_ok=True)

    sf.write(f'{nmf_dir}/{input_name}_vocals.wav', vocals_nmf, sr)
    sf.write(f'{nmf_dir}/{input_name}_acc.wav', acc_nmf, sr)

    print(f"💾 NMF results saved to: {nmf_dir}/")

    # Visualize NMF results
    spec_path = f'{nmf_dir}/{input_name}_spectrogram.png'
    visualize_separation_results(mixture, vocals_nmf, acc_nmf, sr, spec_path)

    # Test 2: Wiener filtering
    print("\n" + "-"*80)
    print("Running: NMF + Wiener Filtering")
    print("-"*80)

    start_time = time.time()
    vocals_wiener, acc_wiener = separator.separate_wiener(mixture, sr)
    time_wiener = time.time() - start_time

    print(f"✅ Completed in {time_wiener:.2f}s")

    # Save Wiener results
    wiener_dir = f'{output_dir}/wiener'
    os.makedirs(wiener_dir, exist_ok=True)

    sf.write(f'{wiener_dir}/{input_name}_vocals.wav', vocals_wiener, sr)
    sf.write(f'{wiener_dir}/{input_name}_acc.wav', acc_wiener, sr)

    print(f"💾 Wiener results saved to: {wiener_dir}/")

    # Visualize Wiener results
    spec_path = f'{wiener_dir}/{input_name}_spectrogram.png'
    visualize_separation_results(mixture, vocals_wiener, acc_wiener, sr, spec_path)

    # Summary
    results = {
        'method_nmf': {
            'time': time_nmf,
            'vocals': vocals_nmf,
            'acc': acc_nmf
        },
        'method_wiener': {
            'time': time_wiener,
            'vocals': vocals_wiener,
            'acc': acc_wiener
        }
    }

    print("\n" + "-"*80)
    print("Timing Summary:")
    print("-"*80)
    print(f"NMF:            {time_nmf:.2f}s")
    print(f"Wiener Filter:  {time_wiener:.2f}s")

    return results


def main():
    """Main function to run all baseline experiments"""

    print("\n" + "="*80)
    print("MUSIC SOURCE SEPARATION - BASELINE EXPERIMENTS")
    print("="*80)

    # Default test file
    default_file = r"C:\Users\17890\Desktop\5305\data\song.mp3"

    # Check if file exists
    if not os.path.exists(default_file):
        print(f"\n⚠️ Default file not found: {default_file}")
        print("   Please provide a valid audio file path.")
        return

    # Run experiments
    try:
        # Test 1: Basic NMF
        test_basic_nmf_separation(default_file)

        # Test 2: Wiener filtering
        test_wiener_filtering(default_file)

        # Test 3: Different configurations
        test_different_nmf_components(default_file)

        # Test 4: Full baseline on real audio
        run_baseline_experiments_real(default_file)

        print("\n" + "="*80)
        print("✅ ALL BASELINE EXPERIMENTS COMPLETED!")
        print("="*80)
        print("\n📂 Results saved to: outputs/")
        print("   - test_basic_nmf/")
        print("   - test_wiener/")
        print("   - test_components/")
        print("   - baseline_real/")

    except Exception as e:
        print(f"\n❌ Error during experiments: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()