import numpy as np, librosa, random

def _safe_pitch_shift(y, sr, n_steps):
    try:
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    except TypeError:
        return librosa.effects.pitch_shift(y, sr, n_steps)

def _safe_time_stretch(y, rate):
    try:
        return librosa.effects.time_stretch(y=y, rate=rate)
    except TypeError:
        return librosa.effects.time_stretch(y, rate)

def _fix_length(y, size):
    try:
        return librosa.util.fix_length(data=y, size=size)
    except TypeError:
        return librosa.util.fix_length(y, size)

def load_audio(path, sr, duration_sec, mono=True):
    y, _ = librosa.load(path, sr=sr, mono=mono)
    target = int(sr * duration_sec)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        y = y[:target]
    return y, sr

def compute_melspec(y, sr, n_mels, n_fft, hop_length, win_length):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       win_length=win_length, n_mels=n_mels, power=2.0)
    S = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    # per-clip standardization
    m = S.mean(); s = S.std() + 1e-8
    return (S - m) / s

def compute_mfcc_stack(y, sr, n_mfcc, n_fft, hop_length, win_length):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                hop_length=hop_length, win_length=win_length)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    X = np.concatenate([mfcc, d1, d2], axis=0).astype(np.float32)
    m = X.mean(); s = X.std() + 1e-8
    return (X - m) / s

def augment(y, sr):
    op = random.choice(["shift","pitch","stretch","noise","none"])
    if op == "shift":
        shift = int(0.05 * sr * random.uniform(-1, 1))
        y = np.roll(y, shift)
    elif op == "pitch":
        y = _safe_pitch_shift(y, sr, n_steps=random.uniform(-2, 2))
    elif op == "stretch":
        y = _safe_time_stretch(y, rate=random.uniform(0.9, 1.1))
    elif op == "noise":
        y = y + 0.01*np.random.randn(len(y)).astype(y.dtype)
    y = _fix_length(y, len(y))  # no-op keeps same len (satisfy older librosa)
    return y
