"""Entrada/saída de áudio (.wav)."""
import numpy as np
import soundfile as sf

import config

config.AUDIO_OUT_DIR.mkdir(exist_ok=True)


def load_audio(path, fs_expected=config.FS):
    """Carrega um .wav como float mono em [-1, 1]. Valida a taxa de amostragem."""
    x, fs = sf.read(str(path), always_2d=False)
    if x.ndim > 1:               # estéreo -> mono
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if fs != fs_expected:
        raise ValueError(f"fs={fs} difere do esperado ({fs_expected}) em {path}")
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak             # normaliza para [-1, 1]
    return x, fs


def save_audio(path, y, fs=config.FS, normalize=True):
    """Salva um sinal em .wav (16 bits), normalizando para evitar clipping."""
    y = np.asarray(y, np.float64)
    if normalize:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
    sf.write(str(path), y.astype(np.float32), fs)
    return path
