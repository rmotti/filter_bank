"""
Runner das Fases 6-7: testes finais.

Teste 1 — fala masculina x feminina (espectrogramas saída x pré-enfatizado).
Teste 2 — float x Q1.15 x Q1.7 (espectrogramas lado a lado + SNR).
Variação do número de canais — 4, 8 e 16.

Gera todos os áudios e figuras de teste e imprime a tabela consolidada de SNR.
"""
import numpy as np

import config
import audio_io
import pipeline
import plots
import metrics

ENTRADAS = {"fem": config.AUDIO_FEM, "masc": config.AUDIO_MASC}
N_LIST = [4, 8, 16]
FORMATS = {"Q1.15": config.Q1_15, "Q1.7": config.Q1_7}


def main():
    print("== Fases 6-7 | Testes (masc/fem, float x fixo, 4/8/16 canais) ==\n")
    tabela = []  # (nome, N, formato, snr)

    for nome, path in ENTRADAS.items():
        x, fs = audio_io.load_audio(path)
        for N in N_LIST:
            # ---- float ----
            y_float, st = pipeline.process(x, n_channels=N, return_stages=True)
            audio_io.save_audio(
                config.AUDIO_OUT_DIR / f"out_{nome}_{N}ch_float.wav", y_float)
            tabela.append((nome, N, "float", float("inf")))

            # Teste 1: saída x pré-enfatizado
            plots.plot_spectrograms(
                [st["xp"], y_float],
                ["Pré-enfatizado", f"Saída IC ({N} canais)"],
                fname=f"08_teste1_{nome}_{N}ch.png",
                sup=f"Teste 1 — {nome}, {N} canais",
            )

            # ---- ponto fixo ----
            specs = [y_float]
            titles = ["float"]
            for nm, fmt in FORMATS.items():
                yq = pipeline.process_fixed(x, fmt, n_channels=N)
                snr = metrics.snr_db(y_float, yq)
                tabela.append((nome, N, nm, snr))
                specs.append(yq)
                titles.append(nm)
                tag = nm.replace(".", "").lower()
                audio_io.save_audio(
                    config.AUDIO_OUT_DIR / f"out_{nome}_{N}ch_{tag}.wav", yq)

            # Teste 2: float x Q1.15 x Q1.7
            plots.plot_spectrograms(
                specs, titles,
                fname=f"09_teste2_{nome}_{N}ch.png",
                sup=f"Teste 2 — {nome}, {N} canais (float x Q1.15 x Q1.7)",
            )
            snrs = {t[2]: t[3] for t in tabela if t[0] == nome and t[1] == N}
            print(f"  {nome:>4} | {N:>2} canais | "
                  f"Q1.15={snrs['Q1.15']:6.2f} dB | Q1.7={snrs['Q1.7']:6.2f} dB")

    # ---- tabela consolidada ----
    print("\n== Tabela consolidada de SNR de saída (dB) ==")
    print("  áudio | N  | Q1.15  | Q1.7")
    print("  ------+----+--------+--------")
    for nome in ENTRADAS:
        for N in N_LIST:
            s = {t[2]: t[3] for t in tabela if t[0] == nome and t[1] == N}
            print(f"  {nome:>4} | {N:>2} | {s['Q1.15']:6.2f} | {s['Q1.7']:6.2f}")
    print("\n(float é a referência; SNR mede a saída fixa contra a saída float)")


if __name__ == "__main__":
    main()
