"""
Etapa 5 — Aritmética de ponto fixo Qm.n.

Simula um DSP/FPGA sem unidade de ponto flutuante: todos os sinais e
coeficientes são inteiros de largura fixa. Ver
docs/03_Tecnico/Modulos/3.6.5_Ponto_Fixo.md.

Convenção: float_to_fixed devolve o INTEIRO que representa o número
(x * 2^n_frac, arredondado e saturado). fixed_to_float faz o caminho inverso.
"""
import numpy as np

import config


# ---------------------------------------------------------------------------
# Parte (a) — Conversão Q-format com saturação
# ---------------------------------------------------------------------------
def _limits(n_bits):
    """Faixa de inteiros representáveis em complemento de dois com n_bits."""
    hi = (1 << (n_bits - 1)) - 1   # ex.: 32767 para 16 bits
    lo = -(1 << (n_bits - 1))      # ex.: -32768 para 16 bits
    return lo, hi


def float_to_fixed(x, n_bits, n_frac):
    """Converte float -> inteiro Qm.n, com arredondamento e SATURAÇÃO.

    Usa apenas operações inteiras após a escala. Aceita escalar ou array.
    """
    lo, hi = _limits(n_bits)
    scaled = np.round(np.asarray(x, dtype=np.float64) * (1 << n_frac))
    # saturação (sem wraparound)
    saturated = np.clip(scaled, lo, hi)
    return saturated.astype(np.int64)


def fixed_to_float(x, n_bits, n_frac):
    """Converte inteiro Qm.n -> float (x / 2^n_frac)."""
    return np.asarray(x, dtype=np.float64) / (1 << n_frac)


def quantize(x, fmt):
    """Quantiza um valor/array float para o formato Qm.n e volta a float.

    Equivale ao erro de representar x naquele formato (round-trip).
    """
    n_bits, n_frac = fmt
    return fixed_to_float(float_to_fixed(x, n_bits, n_frac), n_bits, n_frac)


def quantize_coeffs(coeffs, fmt):
    """Quantiza coeficientes de filtro para o formato Qm.n (retorna float)."""
    return quantize(np.asarray(coeffs, float), fmt)


# ---------------------------------------------------------------------------
# Parte (c) — Convolução FIR em ponto fixo com acumulador de 32 bits
# ---------------------------------------------------------------------------
def fir_fixed(x_int, h_int, n_frac, acc_bits=32, out_bits=16):
    """Convolução FIR inteira com acumulador de `acc_bits` bits.

    x_int e h_int são inteiros Q?.n_frac. Cada saída é o produto-acumulado
    `sum_k x[n-k]*h[k]`, calculado num acumulador de `acc_bits` bits (32);
    ao final, arredonda-se e desloca-se de 2*n_frac de volta para n_frac bits
    fracionários, saturando em `out_bits` (16) bits.

    Implementação vetorizada: a convolução inteira em int64 reproduz exatamente
    o valor do acumulador para todas as amostras de uma vez; em seguida aplicam-se
    saturação de 32 bits, arredondamento/deslocamento e saturação de 16 bits.
    Retorna o sinal de saída como inteiro Q?.n_frac (mesma escala da entrada).
    """
    x_int = np.asarray(x_int, dtype=np.int64)
    h_int = np.asarray(h_int, dtype=np.int64)
    N = len(x_int)
    acc_lo, acc_hi = _limits(acc_bits)
    out_lo, out_hi = _limits(out_bits)
    half = np.int64(1 << (n_frac - 1))  # arredondamento ao reduzir 2n -> n frac

    # acc[n] = sum_k x[n-k]*h[k]  (FIR causal) = primeiros N de 'full'
    acc = np.convolve(x_int, h_int)[:N].astype(np.int64)
    acc = np.clip(acc, acc_lo, acc_hi)        # acumulador de 32 bits
    y = (acc + half) >> n_frac                 # arredonda + desloca p/ n_frac frac
    y = np.clip(y, out_lo, out_hi)             # satura em 16 bits
    return y.astype(np.int64)
