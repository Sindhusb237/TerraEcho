import librosa
import numpy as np

def process_audio(audio_path):

    y, sr = librosa.load(audio_path)

    duration = librosa.get_duration(y=y, sr=sr)

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    rms = np.mean(
        librosa.feature.rms(y=y)
    )

    zero_crossing = np.mean(
        librosa.feature.zero_crossing_rate(y)
    )

    return {
        "y": y,
        "sr": sr,
        "duration": duration,
        "frequency": spectral_centroid,
        "energy": rms,
        "zero_crossing": zero_crossing
    }