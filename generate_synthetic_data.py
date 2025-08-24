# Generates tiny synthetic cough/voice-like WAVs for demo purposes
import os, numpy as np, soundfile as sf, random

def burst_noise(sr, dur, n_bursts=3):
    y = np.zeros(int(sr*dur), dtype=np.float32)
    for _ in range(n_bursts):
        start = random.randint(0, len(y)-int(0.2*sr)-1)
        length = random.randint(int(0.05*sr), int(0.2*sr))
        y[start:start+length] += 0.5*np.random.randn(length).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-6)
    return y

def wheeze_like(sr, dur):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    f = 400 + 200*np.sin(2*np.pi*2*t)  # sweeping
    y = 0.2*np.sin(2*np.pi*f*t).astype(np.float32)
    env = np.exp(-3*t)
    y *= env
    return y

def healthy_voice(sr, dur):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    y = 0.1*np.sin(2*np.pi*180*t) + 0.1*np.sin(2*np.pi*220*t)
    y = y.astype(np.float32)
    return y

def gen_class(out_dir, fn, sr=16000, dur=3.0):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(fn):
        if "Asthma" in out_dir:
            y = wheeze_like(sr, dur)
        elif "Pneumonia" in out_dir:
            y = burst_noise(sr, dur, n_bursts=4)
        else:
            y = healthy_voice(sr, dur)
        sf.write(os.path.join(out_dir, f"sample_{i+1:02d}.wav"), y, sr)

if __name__ == "__main__":
    gen_class("sample_data/Asthma", 5)
    gen_class("sample_data/Pneumonia", 5)
    gen_class("sample_data/Healthy", 5)
    print("Synthetic samples written to sample_data/")
