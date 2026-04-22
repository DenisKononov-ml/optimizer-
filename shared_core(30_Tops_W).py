"""
shared_core(30_Tops_W).py — общая основа для всех оптимизаторов

Содержит:
  - ALL_PARAMS   : полный список параметров с границами
  - BOUNDS       : границы для DE/PSO (list of tuples)
  - compute_objective : единая целевая функция для всех алгоритмов
  - evaluate_metrics_array : интерфейс для DE/PSO (numpy вектор)
  - evaluate_metrics_dict  : интерфейс для CMA-ES/GA (словарь)
  - ConvergenceLogger      : логгер истории сходимости
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

TARGET_CORE_SIZE = 256    # цель по размеру ядра
TARGET_ENERGY    = 200    # точка перегиба энергетического барьера (Вт)

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

ALL_PARAMS = [
    # ── параметры задачи (СВОБОДНЫ) ───────────────────────────────────────────
    # Оптимизируем вместе с чипом — ищем область где OPU максимально выигрывает
    ("num_matrices (n)",    50,       10,         500     ),
    ("batch_size (b)",      1000, 1, 32768    ),
    ("vector_size (v)",     1000, 512, 10000    ),

    # ── параметры чипа (аппаратные) ──────────────────────────────────────────
    ("cores",               1,         1,         128      ),
    ("freq_mhz",            10000,     0.1,       50000    ),
    ("freq_mat_mhz",        500,       0.1,       50000    ),
    ("bw_gbps",             1,         1,         8000     ),
    ("buffer_size_mb",      16,        1,         512      ),
    ("operand_size_bytes",  0.5,       0.5,       0.5      ),  # fix
    ("acc_size_bytes",      4,         4,         4        ),  # fix

    # ── оптические параметры ─────────────────────────────────────────────────
    ("P_laser",             10e-3,     10e-3,     10e-3    ),  # fix
    ("ro_opt",              1,         1,         1        ),  # fix
    ("n_wpe",               0.25,      0.1,       0.25     ),
    ("E_elop",              2e-12,     1e-12,     1e-9     ),
    ("E_elop_driv_vec",     3e-12,     1e-12,     1e-9     ),
    ("E_elop_driv_mat",     3e-12,     1e-12,     1e-9     ),
    ("P_meminterf",         5.77e-3,   2.5e-3,    5.77e-3  ),
    ("P_mat_to",            5e-3,      1e-3,      1.0      ),
    ("E_afe",               2e-12,     1e-12,     1e-9     ),
    ("E_adc_fom",           1000e-15,  22e-15,    20617e-15),
    ("bit",                 4,         4,         4        ),  # fix
    ("n_soa",               3,         0.1,       20       ),
    ("P_soa",               4.2e-2,    4.2e-2,    4.2e-2   ),  # fix
    ("num_soa",             1,         0,         1        ),
    ("B_o",                 2.5e+10,   2.5e+10,   2.5e+10  ),  # fix
    ("P_crit",              10e-3,     10e-3,     10e-3    ),  # fix
    ("IL_splitter",         3.0,       0.1,       10.0     ),
    ("IL_FtoC",             1.0,       0.1,       5.0      ),
    ("IL_SMF",              0.2,       0.01,      1.0      ),
    ("IL_WG",               1.0,       0.1,       5.0      ),
    ("L_MZI_TO",            100e-3,    50e-3,     0.5      ),
    ("L_MZI_EL",            1,         0.5,       2        ),
    ("IL_DC",               0.5,       0.01,      3.0      ),
    ("IL_penalty",          1.0,       0.0,       5.0      ),
    ("IL_to_ps",            1.0,       0.01,      2.8      ),
    ("IL_el_ps",            1.0,       1.0,       5.0      ),
    ("IL_Crossing",         0.05,      0.001,     1.0      ),
    ("RIN",                 -140,      -170,      -100     ),
    ("I_d",                 10e-9,     1e-10,     1e-6     ),
    ("R_PD",                0.5,       0,         1.25     ),
    ("R_l",                 50,        10,        50       ),
]

PARAM_NAMES    = [name for name, *_      in ALL_PARAMS]
INITIAL_GUESS  = [init for _, init, *_   in ALL_PARAMS]
BOUNDS         = [(lo, hi) for _, _, lo, hi in ALL_PARAMS]
BUDGET         = 100000   # используется как справочное значение

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

def compute_objective(energy, core_size, ENOB, performance,
                      energy_eff, Loss) -> tuple:
    """
    Возвращает (value, is_valid).

    Формула: −perf² × eff × barrier_N × barrier_E
      barrier_N — штраф за превышение TARGET_CORE_SIZE (степень 4, резкий)
      barrier_E — мягкий штраф за энергию (степень 2)

    Невалидные точки: (None, False)
    """
    # Проверка на None / NaN / inf
    vals = [energy, core_size, ENOB, performance, energy_eff, Loss]
    if any(v is None for v in vals):
        return None, False
    if any(isinstance(v, float) and (np.isnan(v) or np.isinf(v)) for v in vals):
        return None, False

    # Физические ограничения
    if ENOB < 1.0:
        return None, False
    if performance <= 0.0 or energy_eff <= 0.0:
        return None, False
    if core_size <= 0 or energy <= 0:
        return None, False
    # Нижний порог энергии — менее 1 Вт физически нереалистично
    # для оптического чипа с лазером, АЦП и электроникой управления
    if energy < 1.0:
        return None, False
    # Верхний порог производительности — более 1e6 TOPS нереалистично

    # Барьер по размеру ядра — резкий штраф при core_size > TARGET_CORE_SIZE
    barrier_N = core_size / (
        core_size + 20.0 * max(0.0, core_size - TARGET_CORE_SIZE) ** 4
    )

    # Мягкий барьер по энергии — точка перегиба на TARGET_ENERGY
    # При E=200: barrier_E=0.5, при E=400: barrier_E=0.2, при E=1000: barrier_E=0.04
    barrier_E = 1.0 / (1.0 + (energy / TARGET_ENERGY) ** 2)

    utility = (performance ** 2) * energy_eff * barrier_N * barrier_E

    return -utility, True
