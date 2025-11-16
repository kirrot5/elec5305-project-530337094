"""
NMF-based Music Source Separation with Demucs Enhancement

This module implements several separation methods:
- NMF + Wiener masks (classic single-channel source separation)
- Optional pretrained Demucs (deep learning, if available)
- Simple HPSS-based separation (harmonic/percussive split)

"""

import numpy as np
import librosa
from sklearn.decomposition import NMF
import warnings

warnings.filterwarnings('ignore')

DEMUCS_AVAILABLE = False


def _check_demucs():
    """Check if Demucs is available (lazy import)."""
    global DEMUCS_AVAILABLE
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from demucs.pretrained import get_model  # noqa: F401
        from demucs.apply import apply_model  # noqa: F401
        DEMUCS_AVAILABLE = True
        return True
    except (ImportError, OSError) as e:
        DEMUCS_AVAILABLE = False
        print(f"⚠️ Demucs not available: {e}")
        print("   Using NMF / HPSS only.")
        return False


class NMFSeparator:
    """
    Music source separation using NMF or Demucs.
    Separates mixture into vocals and accompaniment.

    This class illustrates:
    - STFT / iSTFT
    - NMF factorization of magnitude spectrogram
    - Wiener masks (ratio masks) for reconstruction
    """

    def __init__(self,
                 n_components_vocals=30,
                 n_components_acc=50,
                 sr=22050,
                 n_fft=2048,
                 hop_length=512,
                 use_demucs=False):
        """
        Args:
            n_components_vocals: NMF components for vocals
            n_components_acc: NMF components for accompaniment
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            use_demucs: Use pretrained Demucs model if available
                        (default False to avoid torch issues on lab machines)
        """
        # NMF parameters
        self.n_components_vocals = n_components_vocals
        self.n_components_acc = n_components_acc
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Demucs-related
        self.use_demucs = use_demucs
        self.demucs_model = None

        if self.use_demucs:
            if not _check_demucs():
                self.use_demucs = False
                print("   Falling back to NMF...")
            else:
                self._init_demucs()

    # ---------- Demucs (optional deep model) ----------

    def _init_demucs(self):
        """Initialize Demucs model (only called when needed)."""
        try:
            import torch
            from demucs.pretrained import get_model

            print("🔥 Loading Demucs model (htdemucs)...")
            print("   This may take a few seconds on first run...")

            self.demucs_model = get_model('htdemucs')
            self.demucs_model.eval()

            if torch.cuda.is_available():
                self.demucs_model = self.demucs_model.cuda()
                print("✅ Demucs loaded on GPU!")
            else:
                print("✅ Demucs loaded on CPU!")

        except Exception as e:
            print(f"⚠️ Failed to load Demucs: {e}")
            print("   Falling back to NMF...")
            self.use_demucs = False

    def _separate_demucs(self, audio, sr):
        """
        Separate using pretrained Demucs model.

        Note: This is inference-only (no training in this project).
        """
        try:
            import torch
            import torchaudio
            from demucs.apply import apply_model

            # Make stereo
            if audio.ndim == 1:
                audio_stereo = np.stack([audio, audio])
            else:
                audio_stereo = audio

            audio_tensor = torch.from_numpy(audio_stereo).float()

            # Resample to 44.1 kHz as required by Demucs
            if sr != 44100:
                print(f"   Resampling {sr} Hz → 44100 Hz for Demucs...")
                resampler = torchaudio.transforms.Resample(sr, 44100)
                audio_tensor = resampler(audio_tensor)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            audio_tensor = audio_tensor.to(device)

            print("   Separating sources with pretrained Demucs...")
            with torch.no_grad():
                sources = apply_model(
                    self.demucs_model,
                    audio_tensor.unsqueeze(0),
                    split=True,
                    overlap=0.25,
                    device=device
                )[0]

            # Demucs order: [bass, drums, other, vocals]
            vocals_stereo = sources[3].cpu().numpy()
            acc_stereo = sources[[0, 1, 2]].sum(dim=0).cpu().numpy()

            vocals = vocals_stereo.mean(axis=0)
            accompaniment = acc_stereo.mean(axis=0)

            # Resample back to original sr if needed
            if sr != 44100:
                print(f"   Resampling 44100 Hz → {sr} Hz...")
                resampler_back = torchaudio.transforms.Resample(44100, sr)

                vocals_tensor = torch.from_numpy(vocals).float().unsqueeze(0)
                vocals = resampler_back(vocals_tensor).squeeze().numpy()

                acc_tensor = torch.from_numpy(accompaniment).float().unsqueeze(0)
                accompaniment = resampler_back(acc_tensor).squeeze().numpy()

            print("✅ Demucs separation complete!")
            return vocals, accompaniment

        except Exception as e:
            print(f"⚠️ Demucs failed: {e}")
            print("   Falling back to NMF...")
            return self._separate_nmf(audio, sr)

    # ---------- NMF + Wiener masks ----------

    def _separate_nmf(self, audio, sr):
        """
        Separate using Non-negative Matrix Factorization (NMF)
        and Wiener ratio masks (lecture: Wiener Filter & Wiener Masks).

        D(f, t) = STFT{x[n]}
        |D| ≈ W_v H_v + W_a H_a
        masks:
            M_v = S_v / (S_v + S_a)
            M_a = S_a / (S_v + S_a)
        """
        # 1. STFT (time–frequency representation)
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # 2. Independent NMF models for "vocals" and "accompaniment"
        print("   Running NMF decomposition...")

        nmf_vocals = NMF(
            n_components=self.n_components_vocals,
            init='random',
            random_state=0,
            max_iter=200
        )
        nmf_acc = NMF(
            n_components=self.n_components_acc,
            init='random',
            random_state=0,
            max_iter=200
        )

        W_vocals = nmf_vocals.fit_transform(magnitude)
        H_vocals = nmf_vocals.components_
        W_acc = nmf_acc.fit_transform(magnitude)
        H_acc = nmf_acc.components_

        S_vocals = W_vocals @ H_vocals
        S_acc = W_acc @ H_acc

        # 3. Wiener ratio masks (lecture 55–56)
        total = S_vocals + S_acc + 1e-10
        mask_vocals = S_vocals / total
        mask_acc = S_acc / total

        D_vocals = magnitude * mask_vocals * np.exp(1j * phase)
        D_acc = magnitude * mask_acc * np.exp(1j * phase)

        # 4. iSTFT back to time domain
        vocals = librosa.istft(
            D_vocals,
            hop_length=self.hop_length,
            length=len(audio)
        )
        accompaniment = librosa.istft(
            D_acc,
            hop_length=self.hop_length,
            length=len(audio)
        )

        print("✅ NMF + Wiener separation complete!")
        return vocals, accompaniment


    def separate(self, audio, sr=None, method='auto'):
        """
        Separate vocals and accompaniment.

        Args:
            audio: 1D numpy array
            sr: sample rate
            method:
                'auto'   – Demucs if enabled & available, else NMF
                'nmf'    – force NMF
                'demucs' – force Demucs (if available)
        """
        if sr is None:
            sr = self.sr

        if method == 'nmf':
            self.use_demucs = False
        elif method == 'demucs':
            if not self.use_demucs:
                self.use_demucs = _check_demucs()
                if self.use_demucs and self.demucs_model is None:
                    self._init_demucs()

        if self.use_demucs:
            print("🎵 Using Demucs (deep learning)...")
            return self._separate_demucs(audio, sr)
        else:
            print("🎵 Using NMF (traditional, Wiener masks)...")
            return self._separate_nmf(audio, sr)

    def separate_wiener(self, audio, sr=None):
        """Alias kept for backward compatibility."""
        return self.separate(audio, sr, method='nmf')



def hpss_separate(audio, sr, n_fft=2048, hop_length=512):
    """
    Simple harmonic–percussive source separation baseline.

    Uses librosa's HPSS on the STFT:
        S = STFT(x)
        S_harm, S_perc = HPSS(S)

    Not exactly vocals/伴奏, 但演示了另一种基于
    时频结构的分离思路，可作为与 NMF 的对比。
    """
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    # librosa.decompose.hpss 或 librosa.effects.hpss 都可以
    S_harm, S_perc = librosa.decompose.hpss(S)

    harm = librosa.istft(S_harm, hop_length=hop_length, length=len(audio))
    perc = librosa.istft(S_perc, hop_length=hop_length, length=len(audio))
    return harm, perc



def compare_methods(audio_file, output_dir='outputs'):
    """
    Compare NMF and HPSS (and Demucs if available) on the same file.
    This is optional and mainly for exploration.
    """
    import os
    import time
    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📂 Loading: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=22050, mono=True)

    results = {}

    # NMF
    print("\n" + "=" * 50)
    print("Testing NMF")
    print("=" * 50)
    sep_nmf = NMFSeparator(use_demucs=False, sr=sr)
    t0 = time.time()
    vocals_nmf, acc_nmf = sep_nmf.separate(audio, sr, method='nmf')
    time_nmf = time.time() - t0
    sf.write(f'{output_dir}/vocals_nmf.wav', vocals_nmf, sr)
    sf.write(f'{output_dir}/acc_nmf.wav', acc_nmf, sr)
    results['nmf'] = {'time': time_nmf}

    # HPSS
    print("\n" + "=" * 50)
    print("Testing HPSS")
    print("=" * 50)
    t0 = time.time()
    harm, perc = hpss_separate(audio, sr)
    time_hpss = time.time() - t0
    sf.write(f'{output_dir}/harm_hpss.wav', harm, sr)
    sf.write(f'{output_dir}/perc_hpss.wav', perc, sr)
    results['hpss'] = {'time': time_hpss}

    # Demucs (if available)
    if _check_demucs():
        print("\n" + "=" * 50)
        print("Testing Demucs")
        print("=" * 50)
        sep_demucs = NMFSeparator(use_demucs=True, sr=sr)
        t0 = time.time()
        vocals_d, acc_d = sep_demucs.separate(audio, sr, method='demucs')
        time_demucs = time.time() - t0
        sf.write(f'{output_dir}/vocals_demucs.wav', vocals_d, sr)
        sf.write(f'{output_dir}/acc_demucs.wav', acc_d, sr)
        results['demucs'] = {'time': time_demucs}

    print("\n" + "=" * 50)
    print("Summary (runtime only)")
    print("=" * 50)
    print(f"NMF time:   {results['nmf']['time']:.2f} s")
    print(f"HPSS time:  {results['hpss']['time']:.2f} s")
    if 'demucs' in results:
        print(f"Demucs time:{results['demucs']['time']:.2f} s")

    return results
