"""
Runner da Fase 5: implementação em ponto fixo (Q1.15 e Q1.7).

(b) Quantiza coeficientes do banco; resposta original vs Q1.15 vs Q1.7;
    MSE da resposta por canal.
(c) Processa o áudio em Q1.15 e compara com float (SNR, espectrograma).
(d) Repete em Q1.7; monta tabela float x Q1.15 x Q1.7 (SNR, MSE médio de coef.).
"""
import numpy as np

import config
import filters
import audio_io
import pipeline
import plots
import metrics
import fixedpoint as fp

FORMATS = {"Q1.15": config.Q1_15, "Q1.7": config.Q1_7}


def parte_b(n_channels=config.N_CHANNELS):
    print("-- Parte (b): quantização dos coeficientes do banco --\n")
    fc, B = filters.make_channel_table(n_channels)
    bpf = filters.design_filterbank(fc, B)
    bpf_q15 = np.array([fp.quantize_coeffs(h, config.Q1_15) for h in bpf])
    bpf_q7 = np.array([fp.quantize_coeffs(h, config.Q1_7) for h in bpf])

    out = plots.plot_quantized_responses(bpf, bpf_q15, bpf_q7)
    print(f"[gráfico 5] respostas quantizadas -> {out.name}\n")

    # MSE da resposta em frequência por canal (e MSE dos coeficientes)
    print("  canal |  MSE_resp Q1.15 |  MSE_resp Q1.7 |  MSE_coef Q1.15 |  MSE_coef Q1.7")
    print("  ------+-----------------+----------------+-----------------+---------------")
    rmse15 = rmse7 = 0.0
    for k in range(n_channels):
        _, m0 = plots.freq_response(bpf[k])
        _, m15 = plots.freq_response(bpf_q15[k])
        _, m7 = plots.freq_response(bpf_q7[k])
        r15 = metrics.response_mse(m0, m15)
        r7 = metrics.response_mse(m0, m7)
        c15 = metrics.coeff_mse(bpf[k], bpf_q15[k])
        c7 = metrics.coeff_mse(bpf[k], bpf_q7[k])
        rmse15 += c15; rmse7 += c7
        print(f"  {k+1:>5} | {r15:>15.3e} | {r7:>14.3e} | {c15:>15.3e} | {c7:>13.3e}")
    print(f"\n  MSE médio de coeficientes: Q1.15={rmse15/n_channels:.3e}  "
          f"Q1.7={rmse7/n_channels:.3e}")
    return bpf, bpf_q7


def parte_cd(nome="fem", n_channels=config.N_CHANNELS):
    print(f"\n-- Partes (c)/(d): processamento em ponto fixo ({nome}) --\n")
    path = config.AUDIO_FEM if nome == "fem" else config.AUDIO_MASC
    x, fs = audio_io.load_audio(path)

    y_float = pipeline.process(x, n_channels=n_channels)
    audio_io.save_audio(config.AUDIO_OUT_DIR / f"out_{nome}_{n_channels}ch_float.wav", y_float)

    resultados = {"float": (y_float, float("inf"))}
    specs = [y_float]
    titles = ["ponto flutuante"]
    for nm, fmt in FORMATS.items():
        yq = pipeline.process_fixed(x, fmt, n_channels=n_channels)
        snr = metrics.snr_db(y_float, yq)
        resultados[nm] = (yq, snr)
        specs.append(yq)
        titles.append(nm)
        tag = nm.replace(".", "").lower()
        audio_io.save_audio(config.AUDIO_OUT_DIR / f"out_{nome}_{n_channels}ch_{tag}.wav", yq)
        print(f"  {nm}: SNR de saída = {snr:6.2f} dB")

    out = plots.plot_spectrograms(
        specs, titles, fname=f"07_espectro_fixo_{nome}_{n_channels}ch.png",
        sup=f"{nome} — float x Q1.15 x Q1.7",
    )
    print(f"\n[gráfico 6] espectrogramas float x Q1.15 x Q1.7 -> {out.name}")
    return resultados


def main():
    print("== Fase 5 | Ponto fixo ==\n")
    parte_b()
    res = parte_cd("fem")

    print("\n== Tabela comparativa (fem, 8 canais) ==")
    print("  formato     | SNR saída (dB)")
    print("  ------------+---------------")
    for nm in ["float", "Q1.15", "Q1.7"]:
        snr = res[nm][1]
        s = "  (ref)" if np.isinf(snr) else f"{snr:.2f}"
        print(f"  {nm:<11} | {s}")


if __name__ == "__main__":
    main()
