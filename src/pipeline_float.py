"""
Runner da Fase 4: pipeline float completo + áudio reconstruído.

Processa os dois áudios de teste, salva os .wav de saída e gera o
espectrograma comparativo (sinal pré-enfatizado x saída final).
"""
import config
import audio_io
import pipeline
import plots

ENTRADAS = {"fem": config.AUDIO_FEM, "masc": config.AUDIO_MASC}


def main(n_channels=config.N_CHANNELS):
    print(f"== Fase 4 | Pipeline float | N={n_channels} canais ==\n")
    for nome, path in ENTRADAS.items():
        x, fs = audio_io.load_audio(path)
        y, st = pipeline.process(x, n_channels=n_channels, return_stages=True)

        out_wav = config.AUDIO_OUT_DIR / f"out_{nome}_{n_channels}ch_float.wav"
        audio_io.save_audio(out_wav, y)

        fig = plots.plot_spectrograms(
            [st["xp"], y],
            ["Pré-enfatizado (entrada do banco)", "Saída final (IC simulado)"],
            fname=f"05_espectro_{nome}_{n_channels}ch.png",
            sup=f"{nome} — {n_channels} canais",
        )
        print(f"  [{nome}] {len(x)} amostras ({len(x)/fs:.2f}s)")
        print(f"         áudio -> {out_wav.name}")
        print(f"         espectro -> {fig.name}\n")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else config.N_CHANNELS
    main(n)
