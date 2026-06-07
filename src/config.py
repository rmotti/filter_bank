"""
Parâmetros globais do projeto (fonte única de verdade do código).

Espelha docs/03_Tecnico/3.2_Parametros_e_Sinais.md. Qualquer outro módulo
deve importar daqui em vez de repetir valores.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "audios-teste-av3"
AUDIO_FEM = AUDIO_DIR / "vozfeminina.wav"
AUDIO_MASC = AUDIO_DIR / "vozmasculina.wav"

FIG_DIR = ROOT / "figs"          # figuras geradas (ignoradas pelo git)
AUDIO_OUT_DIR = ROOT / "audio_out"  # áudios processados (ignorados pelo git)

# ---------------------------------------------------------------------------
# Parâmetros de sinal
# ---------------------------------------------------------------------------
FS = 22050          # frequência de amostragem (Hz) — confirmada nos áudios de teste
N_CHANNELS = 8      # número de canais padrão (também testamos 4 e 16)

# ---------------------------------------------------------------------------
# Tabela 1 oficial (banco de 8 canais), transcrita do enunciado
#   fc = frequência central (Hz) ; B = largura de banda (Hz)
# ---------------------------------------------------------------------------
TABELA1_FC = [394, 692, 1064, 1528, 2109, 2834, 3740, 4871]
TABELA1_B = [265, 331, 413, 516, 645, 805, 1006, 1257]

# ---------------------------------------------------------------------------
# Decisões de projeto (a justificar no relatório) — valores iniciais
# ---------------------------------------------------------------------------
FIR_LENGTH = 301    # L: comprimento (ímpar) comum a todos os FIR.
#                     Escolhido por varredura: soma do banco fica ±2% de 1.0 no
#                     interior da faixa (os vales de ~0.5 nas bordas extremas são
#                     inerentes ao corte -6 dB do 1o/8o filtro). Ver Fase 2.
NOTCH_BANDWIDTH_HZ = 100.0   # banda alvo do notch DC (<= 100 Hz)
NOTCH_A = 0.99      # polo do notch DC. Menor `a` (entre 0.95/0.98/0.99/0.995)
#                     cuja borda |H|>=0.9 fica <= 100 Hz (a=0.99 -> ~71 Hz). Ver Fase 3.

# Formatos de ponto fixo Qm.n -> (n_bits_total, n_bits_fracionarios)
Q1_15 = (16, 15)
Q1_7 = (8, 7)
