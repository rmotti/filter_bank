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
