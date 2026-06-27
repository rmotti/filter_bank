"""Métricas de comparação: SNR de quantização e MSE."""
import numpy as np


def mse(ref, test):
    """Erro quadrático médio entre dois sinais (alinhados)."""
    ref = np.asarray(ref, float)
    test = np.asarray(test, float)
    n = min(len(ref), len(test))
    return float(np.mean((ref[:n] - test[:n]) ** 2))


def snr_db(ref, test):
    """SNR de quantização (dB): 10·log10( P_sinal / P_ruído ).

    ref = sinal de referência (ponto flutuante); test = versão quantizada.
    O ruído é a diferença test - ref.
    """
    ref = np.asarray(ref, float)
    test = np.asarray(test, float)
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    p_sig = np.sum(ref ** 2)
    p_noise = np.sum((ref - test) ** 2)
    if p_noise == 0:
        return float("inf")
    return float(10.0 * np.log10(p_sig / p_noise))


def coeff_mse(h, h_q):
    """MSE entre os coeficientes originais e quantizados de um filtro."""
    return mse(h, h_q)


def response_mse(mag, mag_q):
    """MSE entre as magnitudes da resposta em frequência original e quantizada."""
    return mse(mag, mag_q)
