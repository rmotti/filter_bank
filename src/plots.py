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


def plot_quantized_responses(bpf, bpf_q15, bpf_q7, fname="06_quant_resp.png",
                            fs=config.FS):
    """Gráfico obrigatório 5: resposta do banco original vs Q1.15 vs Q1.7,
    por canal (grade de subplots)."""
    n = len(bpf)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             squeeze=False)
    for k in range(n):
        ax = axes[k // cols][k % cols]
        f, m0 = freq_response(bpf[k], fs=fs)
        _, m15 = freq_response(bpf_q15[k], fs=fs)
        _, m7 = freq_response(bpf_q7[k], fs=fs)
        ax.plot(f, m0, "k", lw=1.5, label="original")
        ax.plot(f, m15, "tab:blue", lw=1.0, label="Q1.15")
        ax.plot(f, m7, "tab:red", lw=1.0, alpha=0.8, label="Q1.7")
        ax.set_title(f"canal {k+1}", fontsize=9)
        ax.set_xlim(0, fs / 2)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Resposta dos filtros: original vs Q1.15 vs Q1.7")
    fig.tight_layout()
    out = config.FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_spectrograms(signals, titles, fname, fs=config.FS, sup=None):
    """Espectrogramas lado a lado (lista de sinais + títulos)."""
    from scipy.signal import spectrogram
    n = len(signals)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, sig, ttl in zip(axes[0], signals, titles):
        f, t, Sxx = spectrogram(sig, fs=fs, nperseg=512, noverlap=384)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
        ax.set_title(ttl)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência (Hz)")
        ax.set_ylim(0, fs / 2)
    if sup:
        fig.suptitle(sup)
    fig.tight_layout()
    out = config.FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_notch_responses(candidates, a_sel, edges, fname="03_notch.png",
                         fs=config.FS):
    """Gráfico obrigatório 3: notch DC p/ vários `a`, região 0 <= ŵ <= 0.25."""
    import envelope
    w = np.linspace(1e-4, 0.25, 4000)  # frequência normalizada ŵ (rad/amostra)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for a in candidates:
        mag = envelope.notch_magnitude(a, w)
        edge_hz = edges[a]
        sel = " (escolhido)" if a == a_sel else ""
        ax.plot(w, mag, lw=1.6, label=f"a={a}  | borda≈{edge_hz:.0f} Hz{sel}")
    ax.axhline(0.9, color="gray", ls=":", lw=1.0)
    ax.set_title("Notch DC  H(z)=(1-z⁻¹)/(1-a·z⁻¹)  —  |H| em 0 ≤ ŵ ≤ 0.25")
    ax.set_xlabel("ŵ (rad/amostra)")
    ax.set_ylabel("|H(e^{jŵ})|")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = config.FIG_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_channel_cascade(smoother, notch_b, notch_a, canal, fname="04_cascata.png",
                        fs=config.FS):
    """Gráfico obrigatório 4: suavizador, notch e cascata de um canal."""
    f_s, m_s = freq_response(smoother, fs=fs)
    f_n, m_n = freq_response(notch_b, notch_a, fs=fs)
    # cascata: convolução do FIR suavizador com o numerador do notch / denom
    f_c, m_c = freq_response(np.convolve(smoother, notch_b), notch_a, fs=fs)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(f_s, m_s, lw=1.5, label="suavizador (passa-baixa)")
    ax.plot(f_n, m_n, lw=1.5, label="notch DC")
    ax.plot(f_c, m_c, "k", lw=2.0, label="cascata")
    ax.set_title(f"Canal {canal}: suavizador, notch DC e cascata")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("|H(f)|")
    ax.set_xlim(0, fs / 2)
    ax.legend(fontsize=8)
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
