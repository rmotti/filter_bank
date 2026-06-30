"""
Runner da Fase 3: extração de envoltória (suavizador + notch DC).

- Analisa o notch para a = 0.95/0.98/0.99/0.995 (borda onde |H|>=0.9).
- Escolhe `a` para banda do notch <= 100 Hz.
- Gera gráficos obrigatórios 3 (notch) e 4 (cascata de um canal).
"""
import numpy as np

import config
import filters
import envelope
import plots

CANDIDATES = (0.95, 0.98, 0.99, 0.995)
CANAL_ESCOLHIDO = 4  # canal usado na verificação visual da parte (d)


def main():
    print("== Fase 3 | Extração de envoltória ==\n")

    # --- Parte (b)/(c): análise do notch ---
    a_sel, edges = envelope.choose_notch_a(config.NOTCH_BANDWIDTH_HZ,
                                           candidates=CANDIDATES)
    print("  Notch DC — borda de banda (|H| >= 0.9):")
    print("    a      | ŵ_a (rad) | f_a (Hz) | banda <= 100 Hz?")
    print("    -------+-----------+----------+-----------------")
    for a in CANDIDATES:
        f_a, w_a = envelope.notch_band_edge_hz(a)
        ok = "sim" if f_a <= config.NOTCH_BANDWIDTH_HZ else "não"
        marca = "  <- escolhido" if a == a_sel else ""
        print(f"    {a:<6} | {w_a:>9.4f} | {f_a:>8.1f} | {ok}{marca}")
    print(f"\n  => a escolhido = {a_sel} (menor a com banda <= "
          f"{config.NOTCH_BANDWIDTH_HZ:.0f} Hz)")

    out3 = plots.plot_notch_responses(CANDIDATES, a_sel, edges)
    print(f"\n[gráfico 3] notch        -> {out3.name}")

    # --- Parte (a) + (d): suavizadores e cascata de um canal ---
    fc, B = filters.make_channel_table(config.N_CHANNELS)
    smoothers = envelope.design_smoothers(B)
    notch_b, notch_a = envelope.design_dc_notch(a_sel)
    out4 = plots.plot_channel_cascade(smoothers[CANAL_ESCOLHIDO - 1],
                                      notch_b, notch_a, CANAL_ESCOLHIDO)
    print(f"[gráfico 4] cascata ch{CANAL_ESCOLHIDO}   -> {out4.name}")
    print(f"\n  Suavizadores: FPB Hamming L={config.FIR_LENGTH}, "
          f"corte = B_k (largura do canal).")

    return a_sel


if __name__ == "__main__":
    main()
