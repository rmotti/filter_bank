"""Utilitários de visualização (respostas em frequência, somas, espectrogramas)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem janela (salva em arquivo)
import matplotlib.pyplot as plt
from scipy.signal import freqz

import config

config.FIG_DIR.mkdir(exist_ok=True)


def freq_response(b, a=1.0, fs=config.FS, worN=8192):
    """Retorna (f_hz, |H|) da resposta em frequência de um filtro."""
    w, h = freqz(b, a, worN=worN)
    f = w * fs / (2 * np.pi)
    return f, np.abs(h)


def plot_preemphasis(b, a, fname="01_preenfase.png"):
    """Gráfico obrigatório 1: resposta em frequência da pré-ênfase."""
    f, mag = freq_response(b, a)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f, mag, lw=1.8)
    ax.set_title("Resposta em frequência - Pré-ênfase (H(z) = 1 - z⁻¹)")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("|H(f)|")
    ax.set_xlim(0, config.FS / 2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = config.FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_filterbank(taps, fname="02_banco_e_soma.png", fs=config.FS):
    """Gráfico obrigatório 2: N respostas do banco + soma das magnitudes."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    soma = None
    for k, b in enumerate(taps):
        f, mag = freq_response(b, fs=fs)
        ax.plot(f, mag, lw=1.0, alpha=0.8, label=f"canal {k+1}")
        soma = mag if soma is None else soma + mag
    ax.plot(f, soma, "k--", lw=2.0, label="soma")
    ax.axhline(1.0, color="red", ls=":", lw=1.0, alpha=0.7)
    ax.set_title(f"Banco de filtros ({len(taps)} canais) e soma das respostas")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("|H(f)|")
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = config.FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, (f, soma)
