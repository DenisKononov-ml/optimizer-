"""
shared_core.py — общая основа для всех оптимизаторов

Содержит:
  - ALL_PARAMS   : полный список параметров с границами
  - BOUNDS       : границы для DE/PSO (list of tuples)
  - compute_objective : единая целевая функция для всех алгоритмов
  - evaluate_metrics_array : интерфейс для DE/PSO (numpy вектор)
  - evaluate_metrics_dict  : интерфейс для CMA-ES/GA (словарь)
"""

import numpy as np
import yaml
from pathlib import Path

from devices import OPU, GPU, TPU
from calculate import roofline_model
from Params import model_params, model_parts
from Loss_Energy_funcs import defenitive_energy_and_size

# ─────────────────────────────────────────────────────────────────────────────
# Физические константы
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Загрузка GPU и TPU для roofline_model
# ─────────────────────────────────────────────────────────────────────────────

with open(Path("configs/GPU.yaml"), "r", encoding="utf-8") as f:
    model_gpu = GPU(**yaml.safe_load(f)["H100-SXM"])

with open(Path("configs/TPU.yaml"), "r", encoding="utf-8") as f:
    model_tpu = TPU(**yaml.safe_load(f)["V5E-1"])

# ─────────────────────────────────────────────────────────────────────────────
# Параметры задачи и чипа
#
# Формат: (имя, начальное_значение, нижняя_граница, верхняя_граница)
# Если lo == hi — параметр фиксирован, алгоритмы его не трогают.
#
# num_matrices, batch_size, vector_size — СВОБОДНЫ.
# Мы ищем область задач где OPU максимально выигрывает у GPU и TPU.
# Оптимизатор сам найдёт какие задачи выгодны для OPU.
# ─────────────────────────────────────────────────────────────────────────────

# ── Оптимальные значения из лучшего GA эксперимента (используются как init) ──
X_OPT_DICT = {
    "num_matrices (n)": 20.0, "batch_size (b)": 65536.0, "vector_size (v)": 19200.0,
    "cores": 3.0, "freq_mhz": 50000.0, "freq_mat_mhz": 32.77068,
    "bw_gbps": 7811.714463, "buffer_size_mb": 464.990016,
    "n_wpe": 0.25, "E_elop": 1e-12, "E_elop_driv_vec": 1e-12, "E_elop_driv_mat": 1e-12,
    "P_meminterf": 0.0025, "P_mat_to": 0.456864, "E_afe": 1e-12, "E_adc_fom": 22e-15,
    "n_soa": 20.0, "num_soa": 1.0, "IL_splitter": 0.178834, "IL_FtoC": 1.069749,
    "IL_SMF": 0.297142, "IL_WG": 0.119361, "L_MZI_TO": 0.055722, "L_MZI_EL": 0.651149,
    "IL_DC": 0.060375, "IL_penalty": 2.221108, "IL_to_ps": 2.356285, "IL_el_ps": 1.138553,
    "IL_Crossing": 0.001, "RIN": -142.074434, "I_d": 1e-6, "R_PD": 1.027257, "R_l": 26.39929,
}

ALL_PARAMS = [
    # ── параметры задачи ─────────────────────────────────────────────────────
    # init = лучшие значения из GA экспериментов
    ("num_matrices (n)",    20,        1,          10000    ),  # расширен на порядок
    ("batch_size (b)",      65536,     1,          65536    ),  # аппаратный max
    ("vector_size (v)",     19200,     512,        32768    ),  # оригинальные границы

    # ── параметры чипа — расширены на порядок ────────────────────────────────
    ("cores",               3,         1,          1280     ),  # было [1,128]
    ("freq_mhz",            50000,     0.1,        500000   ),  # было [0.1,50000]
    ("freq_mat_mhz",        32.77,     0.1,        500000   ),  # было [0.1,50000]
    ("bw_gbps",             7812,      1,          80000    ),  # было [1,8000]
    ("buffer_size_mb",      450,       1,          5120     ),  # было [1,512]
    ("operand_size_bytes",  0.5,       0.5,        0.5      ),  # fix
    ("acc_size_bytes",      4,         4,          4        ),  # fix

    # ── оптические параметры — ОРИГИНАЛЬНЫЕ границы (модель нестабильна вне них)
    ("P_laser",             10e-3,     10e-3,      10e-3    ),  # fix
    ("ro_opt",              1,         1,          1        ),  # fix
    ("n_wpe",               0.25,      0.1,        0.25     ),  # оригинал
    ("E_elop",              1e-12,     1e-13,      1e-9     ),  # оригинал
    ("E_elop_driv_vec",     1e-12,     1e-13,      1e-9     ),  # оригинал
    ("E_elop_driv_mat",     1e-12,     1e-13,      1e-9     ),  # оригинал
    ("P_meminterf",         2.5e-3,    2.5e-4,     5.77e-3  ),  # оригинал
    ("P_mat_to",            0.457,     1e-4,       1.0      ),  # оригинал
    ("E_afe",               1e-12,     1e-13,      1e-9     ),  # оригинал
    ("E_adc_fom",           22e-15,    22e-16,     20617e-15),  # оригинал
    ("bit",                 4,         4,          4        ),  # fix
    ("n_soa",               20,        0.1,        20       ),  # оригинал
    ("P_soa",               4.2e-2,    4.2e-2,     4.2e-2   ),  # fix
    ("num_soa",             1,         0,          1        ),
    ("B_o",                 2.5e+10,   2.5e+10,    2.5e+10  ),  # fix
    ("P_crit",              10e-3,     10e-3,      10e-3    ),  # fix
    ("IL_splitter",         0.179,     0.1,        10.0     ),  # оригинал
    ("IL_FtoC",             1.07,      0.1,        5.0      ),  # оригинал
    ("IL_SMF",              0.297,     0.01,       1.0      ),  # оригинал
    ("IL_WG",               0.119,     0.1,        5.0      ),  # оригинал
    ("L_MZI_TO",            0.056,     50e-3,      0.5      ),  # оригинал
    ("L_MZI_EL",            0.651,     0.5,        2        ),  # оригинал
    ("IL_DC",               0.060,     0.01,       3.0      ),  # оригинал
    ("IL_penalty",          2.221,     0.0,        5.0      ),  # оригинал
    ("IL_to_ps",            2.356,     0.01,       2.8      ),  # оригинал
    ("IL_el_ps",            1.139,     1.0,        5.0      ),  # оригинал
    ("IL_Crossing",         0.001,     0.001,      1.0      ),  # оригинал
    ("RIN",                 -142,      -170,       -100     ),  # оригинал
    ("I_d",                 1e-6,      1e-10,      1e-6     ),  # оригинал
    ("R_PD",                1.027,     0,          1.25     ),  # оригинал
    ("R_l",                 26.4,      10,         50       ),  # оригинал
]

PARAM_NAMES    = [name for name, *_      in ALL_PARAMS]
INITIAL_GUESS  = [init for _, init, *_   in ALL_PARAMS]
BOUNDS         = [(lo, hi) for _, _, lo, hi in ALL_PARAMS]
BUDGET         = 100000

# Инициализируем X_OPT после определения ALL_PARAMS
X_OPT = np.array([X_OPT_DICT.get(name, init) for name, init, *_ in ALL_PARAMS])

# ─────────────────────────────────────────────────────────────────────────────
# Вычисление физики чипа
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(vals: dict) -> tuple:
    """
    Принимает словарь {имя_параметра: значение}.
    Возвращает (energy, core_size, ENOB, performance, energy_eff, Loss).
    """
    g = vals.get

    n   = g("num_matrices (n)")
    b   = g("batch_size (b)")
    v   = g("vector_size (v)")

    cores          = int(round(g("cores")))
    freq_mhz       = float(g("freq_mhz"))
    freq_mat_mhz   = float(g("freq_mat_mhz"))
    bw_gbps        = float(g("bw_gbps"))
    buffer_size_mb = float(g("buffer_size_mb"))
    operand_size_bytes = float(g("operand_size_bytes"))
    acc_size_bytes     = float(g("acc_size_bytes"))

    safe_bit     = int(round(g("bit")))
    safe_num_soa = int(round(g("num_soa")))
    safe_MZM     = "Thermo" if freq_mat_mhz < 0.5 else "Electro"

    m_params = model_params(
        P_laser          = float(g("P_laser")),
        ro_opt           = float(g("ro_opt")),
        n_wpe            = float(g("n_wpe")),
        E_elop           = float(g("E_elop")),
        DR               = freq_mhz * 1e6,
        matrix_dr        = freq_mat_mhz * 1e6,
        E_elop_driv_vec  = float(g("E_elop_driv_vec")),
        E_elop_driv_mat  = float(g("E_elop_driv_mat")),
        P_meminterf      = float(g("P_meminterf")),
        P_mat_to         = float(g("P_mat_to")),
        E_afe            = float(g("E_afe")),
        E_adc_fom        = float(g("E_adc_fom")),
        bit              = safe_bit,
        MZM              = safe_MZM,
        n_soa            = float(g("n_soa")),
        P_soa            = float(g("P_soa")),
        num_soa          = safe_num_soa,
        B_o              = float(g("B_o")),
        P_crit           = float(g("P_crit")),
        IL_splitter      = float(g("IL_splitter")),
        IL_FtoC          = float(g("IL_FtoC")),
        IL_SMF           = float(g("IL_SMF")),
        IL_WG            = float(g("IL_WG")),
        L_MZI_To         = float(g("L_MZI_TO")),
        L_MZI_El         = float(g("L_MZI_EL")),
        IL_DC            = float(g("IL_DC")),
        IL_penalty       = float(g("IL_penalty")),
        IL_to_ps         = float(g("IL_to_ps")),
        IL_el_ps         = float(g("IL_el_ps")),
        IL_Crossing      = float(g("IL_Crossing")),
        RIN              = float(g("RIN")),
        I_d              = float(g("I_d")),
        R_PD             = float(g("R_PD")),
        R_l              = float(g("R_l")),
    )

    m_parts = model_parts(True, True, True, True, True, True, True, True)
    energy, core_size, ENOB, Loss = defenitive_energy_and_size(
        "Crossbar", m_parts, m_params
    )

    from devices import Task
    task_model = Task(
        num_matrices = int(round(n)),
        batch_size   = int(round(b)),
        vector_size  = int(round(v)),
        matrix_rows  = int(round(v)),
    )

    model_opu = OPU(
        cores              = cores,
        dataflow           = "finite_cache_simple",
        freq_mhz           = freq_mhz,
        freq_mat_mhz       = freq_mat_mhz,
        bw_gbps            = bw_gbps,
        buffer_size_mb     = buffer_size_mb,
        core_size          = core_size,
        acc_size_bytes     = acc_size_bytes,
        operand_size_bytes = operand_size_bytes,
    )

    _, _, _, opu_time, _ = roofline_model(
        int(round(n)), int(round(b)), int(round(v)),
        model_gpu, model_tpu, model_opu,
    )

    total_ops  = 2 * int(round(n)) * int(round(b)) * int(round(v)) * int(round(v))
    performance = (total_ops / opu_time) / 1e12

    # ── Полная энергия: оптика × ядра + SRAM ─────────────────────────────────
    SRAM_POWER_PER_CORE = 1.0   # Вт/МБ
    SRAM_SIZE_PER_CORE  = 16    # МБ на ядро
    sram_power   = SRAM_POWER_PER_CORE * SRAM_SIZE_PER_CORE * cores
    total_energy = energy * cores + sram_power
    energy_eff   = performance / total_energy

    return float(total_energy), float(core_size), float(ENOB), \
           float(performance), float(energy_eff), float(Loss)


def evaluate_metrics_array(x) -> tuple:
    """Интерфейс для DE и PSO — принимает numpy вектор или список."""
    vals = {name: x[i] for i, (name, *_) in enumerate(ALL_PARAMS)}
    return _evaluate(vals)


def evaluate_metrics_dict(params: dict) -> tuple:
    """Интерфейс для CMA-ES и GA — принимает словарь {имя: значение}."""
    # Добавляем фиксированные параметры если их нет в словаре
    vals = {name: init for name, init, *_ in ALL_PARAMS}
    vals.update(params)
    return _evaluate(vals)


# ─────────────────────────────────────────────────────────────────────────────
# Единая целевая функция
# ─────────────────────────────────────────────────────────────────────────────

# ── Параметры новой целевой функции ──────────────────────────────────────────
EFF_TARGET = 100.0   # TOPS/W — пик гауссова вознаграждения
EFF_SIGMA  = 20.0    # ширина пика: ±20 TOPS/W даёт reward > 0.6

# X_OPT как numpy вектор для вычисления drift_penalty


def compute_eff_reward(eff: float) -> float:
    """
    Гауссов пик в EFF_TARGET TOPS/W.
    При eff=100 → 1.0, при eff=80/120 → 0.6, при eff=60/140 → 0.13.
    """
    return float(np.exp(-(eff - EFF_TARGET) ** 2 / (2.0 * EFF_SIGMA ** 2)))


def compute_drift_penalty(x: np.ndarray) -> float:
    """
    Штраф за отклонение от X_OPT.
    Нормировка через x_opt — (x/x_opt - 1) показывает во сколько раз отклонился.
    Возвращает [0, 1] где 1 = нет отклонения.
    """
    xo   = X_OPT
    mask = (np.abs(xo) > 1e-30)  # только параметры с ненулевым x_opt
    if not np.any(mask):
        return 1.0
    rel_sq = ((x[mask] / xo[mask]) - 1.0) ** 2
    return float(1.0 / (1.0 + np.mean(rel_sq)))


def compute_objective(energy, core_size, ENOB, performance,
                      energy_eff, Loss,
                      x: np.ndarray = None) -> tuple:
    """
    Возвращает (value, is_valid).

    Формула: −perf × eff_reward × drift_penalty

      eff_reward   = exp(-(eff - 100)² / (2 × 20²))
                   → гауссов пик в 100 TOPS/W

      drift_penalty = 1 / (1 + mean((x_i/x_opt_i - 1)²))
                    → 1.0 когда параметры у оптимума
                    → снижается при равномерном отходе

    x — numpy вектор параметров (нужен для drift_penalty).
    Если x=None — drift_penalty = 1.0 (нет штрафа).
    """
    vals = [energy, core_size, ENOB, performance, energy_eff, Loss]
    if any(v is None for v in vals):
        return None, False
    if any(isinstance(v, float) and (np.isnan(v) or np.isinf(v)) for v in vals):
        return None, False
    if ENOB < 1.0 or performance <= 0.0 or energy_eff <= 0.0:
        return None, False
    if core_size <= 0 or energy <= 0 or energy < 1.0:
        return None, False

    eff_reward    = compute_eff_reward(energy_eff)
    drift_penalty = compute_drift_penalty(x) if x is not None else 1.0
    barrier_N = core_size / (
        core_size + 20.0 * max(0.0, core_size - 256) ** 4)
    barrier_E = 1.0 / (1.0 + (energy / 200) ** 4)
    utility       = (eff_reward) * drift_penalty * barrier_N * barrier_E

    return -utility, True
