"""
Projeto dos filtros do sistema (parte offline).

Funções que apenas calculam coeficientes — separadas do processamento de
áudio, para que as respostas em frequência possam ser plotadas sem reprocessar
sinal (ver docs/03_Tecnico/3.4_Padroes_de_Codigo.md).
"""
import numpy as np
from scipy.signal import firwin

import config


# ---------------------------------------------------------------------------
# Tabela de canais (fc, B)
# ---------------------------------------------------------------------------
def make_channel_table(n_channels=config.N_CHANNELS):
    """Retorna (fc, B) em Hz para o banco de `n_channels` canais.

    Para N = 8 devolve a Tabela 1 oficial do enunciado. Para outros N, gera
    uma tabela paramétrica mantendo as mesmas propriedades observadas na
    Tabela 1:
      - mesma faixa total coberta (~261.5 Hz a ~5499.5 Hz);
      - larguras de banda em progressão geométrica (escala logarítmica),
        com razão constante r ≈ 1.25 entre bandas adjacentes;
      - canais contíguos (borda superior de um = borda inferior do seguinte).
    """
    if n_channels == 8:
        return np.array(config.TABELA1_FC, float), np.array(config.TABELA1_B, float)

    # Geração paramétrica
    f_low = config.TABELA1_FC[0] - config.TABELA1_B[0] / 2.0   # 261.5 Hz
    f_high = config.TABELA1_FC[-1] + config.TABELA1_B[-1] / 2.0  # 5499.5 Hz
    span = f_high - f_low
    r = 1.25  # razão geométrica das larguras de banda (igual à da Tabela 1)

    # B_1 * (r^N - 1)/(r - 1) = span  ->  B_1
    b1 = span * (r - 1.0) / (r ** n_channels - 1.0)
    widths = b1 * r ** np.arange(n_channels)

    edges = f_low + np.concatenate(([0.0], np.cumsum(widths)))
    fc = (edges[:-1] + edges[1:]) / 2.0
    B = np.diff(edges)
    return fc, B


def band_edges(fc, B):
    """Bordas de banda (corte inferior, superior) de cada canal: fc ± B/2."""
    fc = np.asarray(fc, float)
    B = np.asarray(B, float)
    return fc - B / 2.0, fc + B / 2.0


# ---------------------------------------------------------------------------
# Etapa 1 — Pré-ênfase
# ---------------------------------------------------------------------------
def design_preemphasis():
    """FIR diferenciador de 1a ordem: H(z) = 1 - z^-1.

    Zera totalmente o componente DC (H(z=1) = 0) e realça as altas
    frequências, como pede o enunciado. Retorna (b, a) no padrão scipy.
    """
    b = np.array([1.0, -1.0])
    a = np.array([1.0])
    return b, a


# ---------------------------------------------------------------------------
# Etapa 2 — Banco de filtros passa-banda (Hamming)
# ---------------------------------------------------------------------------
def design_filterbank(fc, B, L=config.FIR_LENGTH, fs=config.FS):
    """Projeta N FIR passa-banda (janela de Hamming), todos de comprimento L.

    Usa as bordas de banda (fc ± B/2) como frequências de corte. Retorna uma
    matriz (N, L) com os coeficientes de cada canal.
    """
    f_low, f_high = band_edges(fc, B)
    nyq = fs / 2.0
    taps = np.zeros((len(fc), L))
    for k in range(len(fc)):
        # firwin normaliza as bordas por Nyquist; pass_zero=False -> passa-banda
        wn = [f_low[k] / nyq, f_high[k] / nyq]
        taps[k] = firwin(L, wn, window="hamming", pass_zero=False)
    return taps
