"""Regras congeladas antes de qualquer resultado do TabPFN.

Mudar um número daqui muda a pergunta. O relatório não escolhe outra regra
depois de ver a AUC.
"""

from __future__ import annotations

PROTOCOL_VERSION = "2"

# O pacote no PyPI, não o default interno (v3.5).
TABPFN_VERSION = "9.2.0"
MODEL_VERSION = "v2"
WEIGHT_REPO = "Prior-Labs/TabPFN-v2-clf"
WEIGHT_FILENAME = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
WEIGHT_SHA256 = "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
WEIGHT_BYTES = 29009539
WHEEL_SHA256 = "bf8edb860cf490ab021d0ba6e91a9798f1b549bdb220066599661c8fc6d6089f"

# Hash dos índices de validação em final_treino_val.csv, semente 42, 5×5.
# Não contém features nem probabilidades.
SPLIT_INDEX_SHA256 = "e896ad31ab4a71dc5b3d4ae13c35cf8008f3276ef391feb8b989de646230d19c"

N_ESTIMATORS = 8
N_SPLITS = 5
N_REPEATS = 5
SEED = 42
N_FITS = N_SPLITS * N_REPEATS
TRAIN_ROWS_PER_FOLD = 5066
SMOKE_ROWS = 48

# Contraste que decide. A média dos três é só relatório.
PRIMARY_CANDIDATE = "tabpfn_v2"
REFERENCE = "ensemble_entregue"
SECONDARY_CANDIDATE = "media_tres"

N_BOOT = 2000
# Estas folgas descrevem o ponto exploratório. Não autorizam adoção.
LR_NEG_WORSE_TOLERANCE = 0.02
F1_WORSE_TOLERANCE = 0.02
BASELINE_ATOL = 1e-4
# CV aninhada de verdade seria 5 grupos externos vezes os 25 fits internos.
NESTED_FIT_COUNT = 5 * 25

CODE_LICENSE = "Apache-2.0"
WEIGHT_LICENSE = "priorlabs-1-1"
WEIGHT_LICENSE_URL = "https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt"
CODE_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
PACKAGE_URL = "https://pypi.org/project/tabpfn/9.2.0/"
HF_MODEL_URL = "https://huggingface.co/Prior-Labs/TabPFN-v2-clf"
HF_FILE_URL = (
    "https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/"
    + WEIGHT_FILENAME
)
S3_FALLBACK_CHECKED = (
    "https://storage.googleapis.com/tabpfn-v2-model-files/05152025/" + WEIGHT_FILENAME
)
S3_FALLBACK_STATUS = "404 em 2026-11-05, não usar"

CPU_SAMPLE_LIMIT_V2 = 1000
V2_MAX_SAMPLES = 10_000
V2_MAX_FEATURES = 500
