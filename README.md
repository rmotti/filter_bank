# Simulação de Implante Coclear com Banco de Filtros — Código

Simulação em software do processador de fala de um implante coclear: pré-ênfase,
banco de filtros FIR (janela de Hamming), extração de envoltória (magnitude +
suavizador + notch DC IIR), modulação senoidal e reimplementação em ponto fixo
(Q1.15 e Q1.7).

## Estrutura

```
.
├── src/                  # código-fonte
├── audios-teste-av3/     # áudios de entrada (vozfeminina.wav, vozmasculina.wav)
├── figs/                 # resultados: gráficos gerados
├── audio_out/            # resultados: áudios processados (.wav)
├── requirements.txt
└── README.md
```

### Módulos (`src/`)
- `config.py` — parâmetros canônicos (fs, Tabela 1, formatos Q, caminhos).
- `audio_io.py` — leitura/escrita de áudio.
- `filters.py` — pré-ênfase e projeto do banco de filtros (Hamming).
- `envelope.py` — suavizadores e notch DC IIR.
- `fixedpoint.py` — conversão Q-format, saturação e FIR com acumulador de 32 bits.
- `metrics.py` — SNR e MSE.
- `pipeline.py` — pipeline completo em ponto flutuante e em ponto fixo.
- `plots.py` — geração das figuras.

### Runners (cada um reproduz uma etapa da entrega)
Executar na ordem abaixo (cada script depende apenas dos áudios de entrada):

| Script | Etapa | Saídas |
|---|---|---|
| `preenfase_e_banco.py` | Pré-ênfase + banco de filtros | figs 01–02 |
| `extracao_envoltoria.py` | Extração de envoltória (notch + cascata) | figs 03–04 |
| `pipeline_float.py` | Pipeline float completo | áudios float + figs 05 |
| `ponto_fixo.py` | Ponto fixo: quantização, SNR, MSE | fig 06–07 + áudios Q1.15/Q1.7 |
| `testes_masc_fem_e_quantizacao.py` | Testes 1 e 2 (masc×fem, float×fixo) | figs 08–09 |

## Como rodar

Requer Python 3.9+.

```bash
# 1. criar ambiente e instalar dependências
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. executar os runners (a partir de src/)
cd src
python preenfase_e_banco.py            # opcional: python preenfase_e_banco.py <N_canais> <L>
python extracao_envoltoria.py
python pipeline_float.py               # opcional: python pipeline_float.py <N_canais>
python ponto_fixo.py
python testes_masc_fem_e_quantizacao.py
```

As figuras são geradas em `figs/` e os áudios processados em `audio_out/`.
Os áudios de entrada já acompanham o pacote em `audios-teste-av3/`.
