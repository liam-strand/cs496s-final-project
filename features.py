from scipy import signal
import numpy as np
import librosa


def reduce_noise(audio_signal, sr=16000, noise_reduction_strength=0.5):
    """Apply spectral gating noise reduction."""
    stft = librosa.stft(audio_signal, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    frame_energy = np.mean(magnitude, axis=0)
    noise_threshold = np.percentile(frame_energy, 10)
    noise_frames = magnitude[:, frame_energy <= noise_threshold]

    if noise_frames.size > 0:
        noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
    else:
        noise_profile = np.min(magnitude, axis=1, keepdims=True)

    magnitude_clean = np.maximum(
        magnitude - noise_reduction_strength * noise_profile, 0
    )

    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=512)

    return audio_clean


def extract_cross_correlation_features(
    audio_signal, sr=16000, apply_noise_reduction=True
):
    if apply_noise_reduction:
        audio_signal = reduce_noise(audio_signal, sr)

    audio_signal = librosa.util.normalize(audio_signal)

    seg1 = audio_signal[:, 0]
    seg2 = audio_signal[:, 1]

    cross_corr = signal.correlate(seg1, seg2, mode="full")
    cross_corr = cross_corr / (np.max(np.abs(cross_corr)) + 1e-10)

    peak_idx = np.argmax(cross_corr)
    expected_center = len(cross_corr) // 2
    phase_shift_samples = peak_idx - expected_center
    phase_shift_ms = (phase_shift_samples / sr) * 1000

    cross_corr_center = cross_corr[len(cross_corr) // 4 : 3 * len(cross_corr) // 4]

    features = np.array(
        [
            phase_shift_ms,
            np.max(cross_corr),
            np.std(cross_corr_center),
            np.sum(np.abs(np.diff(cross_corr))),
        ]
    )

    return features


def extract_all_features_with_xcorr(audio_signal, sr=16000):
    left = audio_signal[:, 0]
    right = audio_signal[:, 1]

    def extract_per_channel(audio):
        audio_clean = librosa.util.normalize(audio)

        mfccs = librosa.feature.mfcc(
            y=audio_clean, sr=sr, n_mfcc=7, n_fft=2048, hop_length=512
        )
        mfcc_features = np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])

        rms = librosa.feature.rms(y=audio_clean)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_clean)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_clean, sr=sr)[0]

        basic_features = np.array(
            [
                np.mean(rms),
                np.std(rms),
                np.mean(zcr),
                np.std(zcr),
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
            ]
        )

        xcorr_features = extract_cross_correlation_features(
            audio_signal, sr, apply_noise_reduction=False
        )

        return np.hstack([mfcc_features, basic_features, xcorr_features])

    return np.hstack([extract_per_channel(left), extract_per_channel(right)])
