"""
Runner das Fases 1 e 2: pré-ênfase e banco de filtros.

Gera os gráficos obrigatórios 1 e 2 e imprime um diagnóstico da planura da
soma do banco (critério para escolher L).

Uso:
    python preenfase_e_banco.py            # N = 8, L do config
    python preenfase_e_banco.py 16 301     # N = 16 canais, L = 301
"""
import sys
import numpy as np

import config
import filters
import plots


def main(n_channels=config.N_CHANNELS, L=config.FIR_LENGTH):
    print(f"== Fase 1-2 | N={n_channels} canais | L={L} ==\n")

    # --- Fase 1: pré-ênfase ---
    b_pre, a_pre = filters.design_preemphasis()
    out1 = plots.plot_preemphasis(b_pre, a_pre)
    print(f"[Fase 1] Pré-ênfase  -> {out1.name}")

    # --- Fase 2: banco de filtros ---
    fc, B = filters.make_channel_table(n_channels)
    f_low, f_high = filters.band_edges(fc, B)
    taps = filters.design_filterbank(fc, B, L=L)
    out2, (f, soma) = plots.plot_filterbank(taps, fname=f"02_banco_{n_channels}ch.png")
    print(f"[Fase 2] Banco       -> {out2.name}")

    # Tabela de canais
    print("\n  canal |   fc (Hz) |   B (Hz) |  corte inf |  corte sup")
    print("  ------+-----------+----------+------------+-----------")
    for k in range(n_channels):
        print(f"  {k+1:>5} | {fc[k]:>9.1f} | {B[k]:>8.1f} | {f_low[k]:>10.1f} | {f_high[k]:>10.1f}")

    # Diagnóstico da planura da soma na faixa coberta pelo banco
    mask = (f >= f_low[0]) & (f <= f_high[-1])
    s = soma[mask]
    print(f"\n  Soma na faixa coberta [{f_low[0]:.0f}-{f_high[-1]:.0f} Hz]:")
    print(f"    min={s.min():.3f}  max={s.max():.3f}  média={s.mean():.3f}  "
          f"ripple=±{(s.max()-s.min())/2:.3f}")
    print(f"    desvio máx. de 1.0: {np.max(np.abs(s-1.0)):.3f}")
    print("\n  (objetivo: soma ≈ 1.0 com ripple pequeno; aumente L se ondular)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else config.N_CHANNELS
    L = int(sys.argv[2]) if len(sys.argv) > 2 else config.FIR_LENGTH
    main(n, L)
