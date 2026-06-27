"""
Pipeline completo do simulador de implante coclear (ponto flutuante).

Encadeia: pré-ênfase -> banco de filtros -> extração de envoltória ->
modulação -> soma. Ver docs/03_Tecnico/3.1_Arquitetura.md.
"""
import numpy as np
from scipy.signal import lfilter

import config
import filters
import envelope
import fixedpoint as fp


# ---------------------------------------------------------------------------
# Etapa 4 — Modulação de envoltória e soma dos canais
# ---------------------------------------------------------------------------
def modulate_and_sum(envs, fc, fs=config.FS):
    """Multiplica cada envoltória por uma senoide em fc_k e soma os canais."""
    envs = np.asarray(envs)
    n = np.arange(envs.shape[1])
    y = np.zeros(envs.shape[1])
    for k in range(envs.shape[0]):
        carrier = np.cos(2 * np.pi * fc[k] * n / fs)
        y += envs[k] * carrier
    return y


# ---------------------------------------------------------------------------
# Pipeline completo (float)
# ---------------------------------------------------------------------------
def process(x, n_channels=config.N_CHANNELS, L=config.FIR_LENGTH,
            a=config.NOTCH_A, fs=config.FS, return_stages=False):
    """Processa o áudio x pelo sistema completo em ponto flutuante.

    Retorna y (saída). Se return_stages=True, retorna também um dicionário com
    sinais intermediários úteis para análise (xp pré-enfatizado, bandas, envs).
    """
    # Etapa 1 — pré-ênfase
    b_pre, a_pre = filters.design_preemphasis()
    xp = lfilter(b_pre, a_pre, x)

    # Etapa 2 — banco de filtros
    fc, B = filters.make_channel_table(n_channels)
    bpf = filters.design_filterbank(fc, B, L=L, fs=fs)
    bands = np.array([lfilter(bpf[k], 1.0, xp) for k in range(n_channels)])

    # Etapa 3 — extração de envoltória (magnitude + suavizador + notch DC)
    smoothers = envelope.design_smoothers(B, L=L, fs=fs)
    notch_b, notch_a = envelope.design_dc_notch(a)
    envs = np.array([envelope.extract_envelope(bands[k], smoothers[k],
                                               notch_b, notch_a)
                     for k in range(n_channels)])

    # Etapa 4 — modulação e soma
    y = modulate_and_sum(envs, fc, fs=fs)

    if return_stages:
        return y, {"xp": xp, "fc": fc, "B": B, "bands": bands, "envs": envs}
    return y


# ---------------------------------------------------------------------------
# Pipeline completo em PONTO FIXO (Etapa 5c/5d)
# ---------------------------------------------------------------------------
def _fir_fp(sig_q, h, fmt):
    """Aplica um FIR em ponto fixo: quantiza sinal e coef., convolve (acc 32b),
    devolve o resultado em float quantizado (mesma escala Qm.n)."""
    n_bits, n_frac = fmt
    xi = fp.float_to_fixed(sig_q, n_bits, n_frac)
    hi = fp.float_to_fixed(h, n_bits, n_frac)
    yi = fp.fir_fixed(xi, hi, n_frac)
    return fp.fixed_to_float(yi, 16, n_frac)


def _notch_fp(x_q, a, fmt):
    """Notch IIR y[n] = x[n] - x[n-1] + a*y[n-1] em ponto fixo (recursivo).

    Coeficiente `a` quantizado; cada amostra de saída é re-quantizada para Qm.n.
    """
    n_bits, n_frac = fmt
    a_q = fp.quantize(a, fmt)
    y = np.zeros_like(x_q)
    x_prev = 0.0
    y_prev = 0.0
    for n in range(len(x_q)):
        val = x_q[n] - x_prev + a_q * y_prev
        val = fp.quantize(val, fmt)   # acumulador reduzido a Qm.n a cada passo
        y[n] = val
        x_prev = x_q[n]
        y_prev = val
    return y


def process_fixed(x, fmt, n_channels=config.N_CHANNELS, L=config.FIR_LENGTH,
                  a=config.NOTCH_A, fs=config.FS):
    """Versão em ponto fixo do pipeline completo, no formato Qm.n `fmt`.

    Honra: coeficientes quantizados, convolução FIR com acumulador de 32 bits e
    saturação. Operações não lineares (|.|, notch IIR, modulação) trabalham no
    domínio quantizado. Retorna y (float quantizado).
    """
    n_bits, n_frac = fmt

    # Etapa 1 — pré-ênfase
    b_pre, _ = filters.design_preemphasis()
    xq = fp.quantize(x, fmt)
    xp = _fir_fp(xq, b_pre, fmt)

    # Etapa 2 — banco de filtros
    fc, B = filters.make_channel_table(n_channels)
    bpf = filters.design_filterbank(fc, B, L=L, fs=fs)
    smoothers = envelope.design_smoothers(B, L=L, fs=fs)

    y = np.zeros(len(x))
    for k in range(n_channels):
        band = _fir_fp(xp, bpf[k], fmt)            # Etapa 2
        mag = fp.quantize(np.abs(band), fmt)        # Etapa 3 — magnitude
        smooth = _fir_fp(mag, smoothers[k], fmt)    # Etapa 3 — suavizador
        env = _notch_fp(smooth, a, fmt)             # Etapa 3 — notch DC
        # Etapa 4 — modulação (portadora quantizada) e acumulação
        n = np.arange(len(env))
        carrier = fp.quantize(np.cos(2 * np.pi * fc[k] * n / fs), fmt)
        y += fp.quantize(env * carrier, fmt)
    return y
