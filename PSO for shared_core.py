import sys
import time
import numpy as np

from shared_core import (
    ALL_PARAMS,
    evaluate_metrics_array, compute_objective,
)

SEED        = 70
POP_SIZE    = 30
STALL_LIMIT = 300
MAXITER     = 3000
PRINT_EACH  = 50

# Параметры PSO
W_MAX       = 0.9      # начальный инерционный вес
W_MIN       = 0.4      # конечный инерционный вес
C1          = 2.0      # когнитивный коэффициент (личный лучший)
C2          = 2.0      # социальный коэффициент (глобальный лучший)
V_MAX_FRAC  = 0.2      # макс. скорость = V_MAX_FRAC * (UB - LB)

# Параметры DE-мутации (применяется при стагнации частицы)
F_DE        = 0.5      # масштаб мутации DE
CR_DE       = 0.9      # вероятность скрещивания DE
STALL_DE    = 15       # после скольких итераций без улучшения частицы — DE

# Параметры WOA спирального поиска (поздние итерации)
WOA_START   = 0.6      # доля от MAXITER, с которой включается WOA
B_SPIRAL    = 1.0      # константа спирали WOA

# ─────────────────────────────────────────────────────────────────────────────

INT_PARAMS  = {"cores", "num_soa", "num_matrices (n)", "batch_size (b)", "vector_size (v)"}
PARAM_NAMES = [name for name, *_ in ALL_PARAMS]
IS_INT      = np.array([name in INT_PARAMS for name in PARAM_NAMES], dtype=bool)
LB          = np.array([lo for _, _, lo, hi in ALL_PARAMS], dtype=float)
UB          = np.array([hi for _, _, lo, hi in ALL_PARAMS], dtype=float)

GROUPS = {
    "Параметры задачи":  ["num_matrices (n)", "batch_size (b)", "vector_size (v)"],
    "Параметры чипа":    ["cores", "freq_mhz", "freq_mat_mhz", "bw_gbps", "buffer_size_mb"],
    "Оптические":        ["n_wpe", "E_elop", "E_elop_driv_vec", "E_elop_driv_mat",
                          "P_meminterf", "P_mat_to", "E_afe", "E_adc_fom",
                          "n_soa", "num_soa", "IL_splitter", "IL_FtoC", "IL_SMF",
                          "IL_WG", "L_MZI_TO", "L_MZI_EL", "IL_DC", "IL_penalty",
                          "IL_to_ps", "IL_el_ps", "IL_Crossing", "RIN", "I_d",
                          "R_PD", "R_l"],
}

print(f"NDWPSO-2024 | pop={POP_SIZE} | stall={STALL_LIMIT} | maxiter={MAXITER} | seed={SEED}")
print(f"Улучшения: нелин. инерция + jump-out + WOA-спираль + DE-мутация")
print(f"{'─'*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Целевая функция
# ─────────────────────────────────────────────────────────────────────────────

def objective(x):
    try:
        energy, core_size, ENOB, perf, eff, Loss = evaluate_metrics_array(x)
        val, ok = compute_objective(energy, core_size, ENOB, perf, eff, Loss)
    except Exception:
        return 1e10
    return val if ok else 1e10

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

rng   = np.random.default_rng(SEED)
V_MAX = V_MAX_FRAC * (UB - LB)
V_MAX = np.where(LB == UB, 0.0, V_MAX)

def enforce(x):
    """Приводим координаты к границам и округляем целочисленные."""
    x = np.clip(x, LB, UB)
    x[IS_INT] = np.round(x[IS_INT])
    return x

def init_population():
    """
    Elite Opposition-Based Learning инициализация (EOBL).
    Генерируем POP_SIZE частиц + POP_SIZE отражённых,
    отбираем лучшие POP_SIZE.
    """
    # Случайная популяция (Latin Hypercube)
    pop = np.zeros((POP_SIZE, len(LB)))
    for j in range(len(LB)):
        if LB[j] == UB[j]:
            pop[:, j] = LB[j]
        elif IS_INT[j]:
            pop[:, j] = rng.integers(int(LB[j]), int(UB[j]) + 1, POP_SIZE).astype(float)
        else:
            perm = rng.permutation(POP_SIZE)
            pop[:, j] = LB[j] + (perm + rng.random(POP_SIZE)) / POP_SIZE * (UB[j] - LB[j])

    # Elite opposition: отражённые частицы относительно центра области
    center = (LB + UB) / 2.0
    opp = 2 * center - pop
    opp = np.clip(opp, LB, UB)
    opp[:, IS_INT] = np.round(opp[:, IS_INT])

    # Объединяем и отбираем лучшие POP_SIZE
    combined    = np.vstack([pop, opp])
    fitness_all = np.array([objective(combined[i]) for i in range(2 * POP_SIZE)])
    best_idx    = np.argsort(fitness_all)[:POP_SIZE]
    return combined[best_idx].copy(), fitness_all[best_idx].copy()

def nonlinear_inertia(it):
    """
    Нелинейный убывающий инерционный вес.
    w(t) = W_MIN + (W_MAX - W_MIN) * ((MAXITER - t) / MAXITER)^2
    Быстро убывает в начале (широкий поиск), медленно в конце (точная настройка).
    """
    return W_MIN + (W_MAX - W_MIN) * ((MAXITER - it) / MAXITER) ** 2

def de_mutation(pop, i):
    """
    DE/rand/1/bin мутация для частицы i.
    Применяется когда частица застряла (личный stall > STALL_DE).
    """
    candidates = [j for j in range(POP_SIZE) if j != i]
    r1, r2, r3 = rng.choice(candidates, 3, replace=False)
    mutant = pop[r1] + F_DE * (pop[r2] - pop[r3])
    cross  = rng.random(len(LB)) < CR_DE
    cross[rng.integers(len(LB))] = True
    trial  = np.where(cross, mutant, pop[i])
    return enforce(trial)

def woa_spiral(x, gbest):
    """
    Спиральное обновление позиции из Whale Optimization Algorithm.
    Применяется в поздних итерациях (it > WOA_START * MAXITER).
    x_new = D * e^(b*l) * cos(2π*l) + gbest
    где D = |gbest - x|, l ∈ [-1, 1]
    """
    l     = rng.uniform(-1, 1, len(LB))
    D     = np.abs(gbest - x)
    x_new = D * np.exp(B_SPIRAL * l) * np.cos(2 * np.pi * l) + gbest
    return enforce(x_new)

# ─────────────────────────────────────────────────────────────────────────────
# Инициализация
# ─────────────────────────────────────────────────────────────────────────────

t0      = time.time()
n_calls = 0

print("  Инициализация с EOBL (2×POP оценок)...")
pop, fitness = init_population()
n_calls += 2 * POP_SIZE

# Скорости: равномерно в [-V_MAX, V_MAX]
vel = rng.uniform(-1, 1, (POP_SIZE, len(LB))) * V_MAX

# Личные лучшие
pbest_x = pop.copy()
pbest_f = fitness.copy()

# Глобальный лучший
best_idx = np.argmin(fitness)
gbest_f  = fitness[best_idx]
gbest_x  = pop[best_idx].copy()

# Счётчик стагнации для каждой частицы (для DE)
particle_stall = np.zeros(POP_SIZE, dtype=int)

print(f"  Старт после EOBL: f = {gbest_f:.4e}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Основной цикл NDWPSO
# ─────────────────────────────────────────────────────────────────────────────

stall     = 0
converged = False
it        = 0

for it in range(MAXITER):
    w       = nonlinear_inertia(it)
    use_woa = (it > WOA_START * MAXITER)

    improved_global = False

    for i in range(POP_SIZE):

        # ── Выбор стратегии обновления ──────────────────────────────────────
        if particle_stall[i] >= STALL_DE:
            # DE-мутация для застрявшей частицы
            x_new = de_mutation(pop, i)
            v_new = vel[i]
        elif use_woa and rng.random() < 0.5:
            # WOA-спираль в поздних итерациях (с вероятностью 0.5)
            x_new = woa_spiral(pop[i], gbest_x)
            v_new = vel[i]
        else:
            # Стандартное PSO-обновление
            r1    = rng.random(len(LB))
            r2    = rng.random(len(LB))
            v_new = (w * vel[i]
                     + C1 * r1 * (pbest_x[i] - pop[i])
                     + C2 * r2 * (gbest_x    - pop[i]))
            v_new = np.clip(v_new, -V_MAX, V_MAX)
            x_new = enforce(pop[i] + v_new)

        # ── Оценка ──────────────────────────────────────────────────────────
        f_new    = objective(x_new)
        n_calls += 1

        # ── Обновление личного лучшего ───────────────────────────────────────
        if f_new < pbest_f[i]:
            pbest_f[i]        = f_new
            pbest_x[i]        = x_new.copy()
            particle_stall[i] = 0
        else:
            particle_stall[i] += 1

        pop[i]     = x_new
        vel[i]     = v_new
        fitness[i] = f_new

        # ── Обновление глобального лучшего ───────────────────────────────────
        if f_new < gbest_f:
            gbest_f         = f_new
            gbest_x         = x_new.copy()
            improved_global = True

    # ── Jump-out: перезапуск худших 20% вокруг gbest ────────────────────────
    if not improved_global:
        stall   += 1
        n_worst  = max(1, POP_SIZE // 5)
        worst_idx = np.argsort(fitness)[-n_worst:]
        for idx in worst_idx:
            noise       = rng.uniform(-0.1, 0.1, len(LB)) * (UB - LB)
            pop[idx]    = enforce(gbest_x + noise)
            vel[idx]    = rng.uniform(-1, 1, len(LB)) * V_MAX * 0.5
            fitness[idx] = objective(pop[idx])
            n_calls     += 1
            if fitness[idx] < gbest_f:
                gbest_f = fitness[idx]
                gbest_x = pop[idx].copy()
    else:
        stall = 0

    # ── Лог ─────────────────────────────────────────────────────────────────
    if it % PRINT_EACH == 0:
        mode_str = "WOA+DE" if use_woa else "PSO+DE"
        print(f"  iter {it:4d} | calls {n_calls:7d} | best {gbest_f:.4e} | "
              f"stall {stall:3d}/{STALL_LIMIT} | w={w:.3f} | "
              f"{mode_str} | {time.time()-t0:.1f}s")

    if stall >= STALL_LIMIT:
        converged = True
        break

elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# Итог
# ─────────────────────────────────────────────────────────────────────────────

try:
    energy, core_size, ENOB, perf, eff, Loss = evaluate_metrics_array(gbest_x)
except Exception:
    energy = core_size = ENOB = perf = eff = Loss = None

print(f"\n{'═'*65}")
print(f"  {'Сошёлся' if converged else 'Бюджет исчерпан'} | "
      f"итерация {it} | {elapsed:.1f}s | вызовов {n_calls:,}")
print(f"{'═'*65}")
print(f"  f(x)          = {gbest_f:.9e}")
print(f"  Performance   = {perf:.2f} TOPS"  if perf      else "  Performance  = —")
print(f"  Energy        = {energy:.2f} Вт"  if energy    else "  Energy       = —")
print(f"  Core size N   = {core_size:.1f}"  if core_size else "  Core size    = —")
print(f"  ENOB          = {ENOB:.4f}"        if ENOB      else "  ENOB         = —")
print(f"  Energy eff    = {eff:.4f} TOPS/W" if eff       else "  Energy eff   = —")
print(f"  Loss          = {Loss:.4f}"        if Loss      else "  Loss         = —")

# ─────────────────────────────────────────────────────────────────────────────
# Оптимальные параметры — по группам
# ─────────────────────────────────────────────────────────────────────────────

best_dict = {name: float(gbest_x[i]) for i, name in enumerate(PARAM_NAMES)}

print(f"\n{'─'*65}")
print(f"  ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ")
print(f"{'─'*65}")

for group_name, param_list in GROUPS.items():
    print(f"\n  ┌─ {group_name}")
    for name in param_list:
        if name not in best_dict:
            continue
        val = best_dict[name]
        idx = PARAM_NAMES.index(name)
        lo  = LB[idx]
        hi  = UB[idx]

        if abs(val) < 1e-6 or abs(val) > 1e6:
            val_str = f"{val:.4e}"
        elif IS_INT[idx]:
            val_str = f"{int(val)}"
        else:
            val_str = f"{val:.6g}"

        if hi > lo:
            pct    = (val - lo) / (hi - lo) * 100
            at_max = " ← МАКСИМУМ" if pct > 99 else ""
            at_min = " ← МИНИМУМ"  if pct < 1  else ""
            bounds_str = f"  [{lo:.4g} .. {hi:.4g}]  {pct:.0f}%{at_max}{at_min}"
        else:
            bounds_str = "  [фикс]"

        is_int_str = " (int)" if IS_INT[idx] else ""
        print(f"  │  {name:<22} = {val_str:<14}{is_int_str}{bounds_str}")
