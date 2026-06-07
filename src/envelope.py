"""
Etapa 3 — Extração de envoltória.

Cadeia por canal: |.| (magnitude)  ->  suavizador FPB Hamming  ->  notch DC IIR.
Ver docs/03_Tecnico/Modulos/3.6.3_Extracao_de_Envoltoria.md.
"""
import numpy as np
from scipy.signal import firwin, lfilter

import config


# ---------------------------------------------------------------------------
# Parte (a) — Suavizadores (um FPB passa-baixa por canal)
# ---------------------------------------------------------------------------
def design_smoothers(B, L=config.FIR_LENGTH, fs=config.FS):
    """FPB Hamming passa-baixa por canal, faixa de passagem plana sobre B_k.

    A magnitude |band_k| produz um sinal = DC + envoltória lenta (conteúdo até
    ~B_k) + réplicas em torno de 2*fc_k. O suavizador passa a envoltória
    (corte ~ B_k) e rejeita as réplicas (que ficam bem acima). Retorna (N, L).
    """
    B = np.asarray(B, float)
    nyq = fs / 2.0
    taps = np.zeros((len(B), L))
    for k in range(len(B)):
        taps[k] = firwin(L, B[k] / nyq, window="hamming")  # passa-baixa
    return taps


# ---------------------------------------------------------------------------
# Parte (b)/(c) — Notch IIR para remover DC
# ---------------------------------------------------------------------------
def design_dc_notch(a):
    """Notch DC: H(z) = (1 - z^-1) / (1 - a z^-1). Zero em z=1, polo em z=a."""
    b = np.array([1.0, -1.0])
    a_coeffs = np.array([1.0, -a])
    return b, a_coeffs


def notch_magnitude(a, w):
    """|H(e^jw)| analítico do notch DC para um vetor de frequências w (rad)."""
    num2 = 2.0 - 2.0 * np.cos(w)                 # |1 - e^-jw|^2
    den2 = 1.0 - 2.0 * a * np.cos(w) + a * a     # |1 - a e^-jw|^2
    return np.sqrt(num2 / den2)


def notch_band_edge_hz(a, fs=config.FS, thresh=0.9):
    """Menor frequência (Hz) a partir da qual |H| >= thresh (borda da banda).

    Resolve |H(w)| = thresh para o notch DC. Como |H| cresce monotonicamente
    de 0 (em w=0) até ~1, existe uma única borda.
    """
    t2 = thresh * thresh
    # (2 - 2cosw)/(1 - 2a cosw + a^2) = t2  ->  resolve para cosw
    # 2 - 2cosw = t2 (1 + a^2) - 2 a t2 cosw
    # cosw (2 a t2 - 2) = t2 (1 + a^2) - 2
    cos_w = (t2 * (1 + a * a) - 2.0) / (2.0 * a * t2 - 2.0)
    cos_w = np.clip(cos_w, -1.0, 1.0)
    w = np.arccos(cos_w)
    return w * fs / (2.0 * np.pi), w


def choose_notch_a(target_bw_hz=config.NOTCH_BANDWIDTH_HZ, fs=config.FS,
                   candidates=(0.95, 0.98, 0.99, 0.995)):
    """Entre os candidatos, escolhe o menor `a` cuja banda do notch <= alvo.

    Quanto maior `a` (mais perto de 1), mais estreito o notch. Retorna
    (a_escolhido, dict {a: borda_hz}).
    """
    edges = {a: notch_band_edge_hz(a, fs)[0] for a in candidates}
    validos = [a for a in candidates if edges[a] <= target_bw_hz]
    a_sel = min(validos) if validos else max(candidates)
    return a_sel, edges


# ---------------------------------------------------------------------------
# Cadeia completa de envoltória (por canal)
# ---------------------------------------------------------------------------
def extract_envelope(band, smoother, notch_b, notch_a):
    """|band| -> suavização (passa-baixa) -> remoção de DC (notch). Retorna env."""
    mag = np.abs(band)
    smooth = lfilter(smoother, 1.0, mag)
    env = lfilter(notch_b, notch_a, smooth)
    return env
