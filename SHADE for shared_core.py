import sys
import time
import numpy as np

from shared_core import (
    ALL_PARAMS,
    evaluate_metrics_array, compute_objective,
)

# ─────────────────────────────────────────────────────────────────────────────
# Настройки
# ─────────────────────────────────────────────────────────────────────────────

SEED        = 70
POP_SIZE    = 15
H           = 10
P_BEST      = 0.15
STALL_LIMIT = 150
MAXITER     = 3000
P_NOISE     = 0.15
PRINT_EACH  = 50

INT_PARAMS  = {"cores", "num_soa", "num_matrices (n)", "batch_size (b)", "vector_size (v)"}
PARAM_NAMES = [name for name, *_ in ALL_PARAMS]
IS_INT      = np.array([name in INT_PARAMS for name in PARAM_NAMES], dtype=bool)
LB          = np.array([lo for _, _, lo, hi in ALL_PARAMS], dtype=float)
UB          = np.array([hi for _, _, lo, hi in ALL_PARAMS], dtype=float)

# Группировка для красивого вывода
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

print(f"SHADE | pop={POP_SIZE}×{len(ALL_PARAMS)}={POP_SIZE*len(ALL_PARAMS)} | "
      f"stall={STALL_LIMIT} | seed={SEED}")
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
# SHADE
# ─────────────────────────────────────────────────────────────────────────────

rng       = np.random.default_rng(SEED)
M_F       = np.full(H, 0.5)
M_CR      = np.full(H, 0.5)
k         = 0
archive   = []
n_calls   = 0
gbest_f   = np.inf
gbest_x   = None
converged = False

def enforce(x):
    x = np.clip(x, LB, UB)
    x[IS_INT] = np.round(x[IS_INT])
    return x

def init_population():
    pop = np.zeros((POP_SIZE, len(LB)))
    for j in range(len(LB)):
        if LB[j] == UB[j]:
            pop[:, j] = LB[j]
        elif IS_INT[j]:
            pop[:, j] = rng.integers(int(LB[j]), int(UB[j])+1, POP_SIZE).astype(float)
        else:
            perm = rng.permutation(POP_SIZE)
            pop[:, j] = LB[j] + (perm + rng.random(POP_SIZE)) / POP_SIZE * (UB[j] - LB[j])
    return pop

def sample_F_CR():
    F_arr, CR_arr = np.zeros(POP_SIZE), np.zeros(POP_SIZE)
    for i in range(POP_SIZE):
        r = rng.integers(0, H)
        F = 0.0
        while F <= 0:
            F = min(M_F[r] + 0.1 * rng.standard_cauchy(), 1.0)
        F_arr[i]  = F
        CR_arr[i] = np.clip(M_CR[r] + 0.1 * rng.standard_normal(), 0.0, 1.0)
    return F_arr, CR_arr

def mutate(pop, fitness, F_arr):
    n_pbest    = max(2, int(POP_SIZE * P_BEST))
    pbest_pool = np.argsort(fitness)[:n_pbest]
    union_pop  = np.vstack([pop, np.array(archive[-POP_SIZE:])]) if archive else pop
    trial = np.zeros_like(pop)
    for i in range(POP_SIZE):
        F = F_arr[i]
        r1 = i
        while r1 == i: r1 = rng.integers(0, POP_SIZE)
        r2 = i
        while r2 == i or r2 == r1: r2 = rng.integers(0, len(union_pop))
        trial[i] = pop[i] + F*(pop[rng.choice(pbest_pool)]-pop[i]) + F*(pop[r1]-union_pop[r2])
    return trial

def crossover(pop, trial, CR_arr):
    u = np.empty_like(pop)
    for i in range(POP_SIZE):
        cross = rng.random(len(LB)) < CR_arr[i]
        cross[rng.integers(len(LB))] = True
        u[i] = enforce(np.where(cross, trial[i], pop[i]))
        for j in np.where(IS_INT & (LB < UB))[0]:
            if rng.random() < P_NOISE:
                delta = rng.integers(1, 3) * rng.choice([-1, 1])
                u[i, j] = float(np.clip(u[i, j] + delta, LB[j], UB[j]))
    return u

def update_memory(S_F, S_CR, weights):
    global k
    if not S_F: return
    w = np.array(weights); w = w / w.sum()
    f_arr = np.array(S_F); cr_arr = np.array(S_CR)
    M_F[k]  = np.clip(np.sum(w*f_arr**2)/(np.sum(w*f_arr)+1e-12), 0.01, 1.0)
    M_CR[k] = np.sum(w * cr_arr)
    k = (k + 1) % H

# ─────────────────────────────────────────────────────────────────────────────
# Запуск
# ─────────────────────────────────────────────────────────────────────────────

t0  = time.time()
pop = init_population()
fitness = np.full(POP_SIZE, np.inf)
for i in range(POP_SIZE):
    fitness[i] = objective(pop[i])
    n_calls += 1

best_idx = np.argmin(fitness)
gbest_f  = fitness[best_idx]
gbest_x  = pop[best_idx].copy()
print(f"  Старт: f = {gbest_f:.4e}\n")

stall = 0
for it in range(MAXITER):
    F_arr, CR_arr = sample_F_CR()
    trial = mutate(pop, fitness, F_arr)
    u     = crossover(pop, trial, CR_arr)

    S_F, S_CR, weights = [], [], []
    improved = False
    for i in range(POP_SIZE):
        f_u = objective(u[i])
        n_calls += 1
        if f_u <= fitness[i]:
            archive.append(pop[i].copy())
            S_F.append(F_arr[i]); S_CR.append(CR_arr[i])
            weights.append(max(0.0, fitness[i] - f_u))
            pop[i] = u[i]; fitness[i] = f_u
            rel = abs(f_u - gbest_f) / (abs(gbest_f) + 1e-30)
            if f_u < gbest_f and rel > 1e-6:
                gbest_f = f_u; gbest_x = u[i].copy(); improved = True

    if len(archive) > POP_SIZE * 2:
        archive[:] = archive[-POP_SIZE * 2:]

    if S_F:
        w = weights if any(w > 0 for w in weights) else [1.0/len(S_F)]*len(S_F)
        update_memory(S_F, S_CR, w)

    stall = 0 if improved else stall + 1

    if it % PRINT_EACH == 0:
        print(f"  iter {it:4d} | calls {n_calls:7d} | best {gbest_f:.4e} | "
              f"stall {stall:3d}/{STALL_LIMIT} | "
              f"M_F={M_F.mean():.2f} M_CR={M_CR.mean():.2f} | "
              f"{time.time()-t0:.1f}s")

    if stall >= STALL_LIMIT:
        converged = True
        break

elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# Итог — метрики
# ─────────────────────────────────────────────────────────────────────────────

try:
    energy, core_size, ENOB, perf, eff, Loss = evaluate_metrics_array(gbest_x)
except Exception:
    energy = core_size = ENOB = perf = eff = Loss = None

print(f"\n{'═'*65}")
print(f"  {'Сошёлся' if converged else 'Бюджет исчерпан'} | итерация {it} | {elapsed:.1f}s | вызовов {n_calls:,}")
print(f"{'═'*65}")
print(f"  f(x)          = {gbest_f:.9e}")
print(f"  Performance   = {perf:.2f} TOPS"  if perf      else "  Performance  = —")
print(f"  Energy        = {energy:.2f} Вт"  if energy    else "  Energy       = —")
print(f"  Core size N   = {core_size:.1f}"  if core_size else "  Core size    = —")
print(f"  ENOB          = {ENOB:.4f}"        if ENOB      else "  ENOB         = —")
print(f"  Energy eff    = {eff:.4f} TOPS/W" if eff       else "  Energy eff   = —")
print(f"  Loss          = {Loss:.4f}"        if Loss      else "  Loss         = —")

# ─────────────────────────────────────────────────────────────────────────────
# Все оптимальные параметры — по группам
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
        lo  = LB[PARAM_NAMES.index(name)]
        hi  = UB[PARAM_NAMES.index(name)]

        if abs(val) < 1e-6 or abs(val) > 1e6:
            val_str = f"{val:.4e}"
        elif IS_INT[PARAM_NAMES.index(name)]:
            val_str = f"{int(val)}"
        else:
            val_str = f"{val:.6g}"

        if hi > lo:
            pct = (val - lo) / (hi - lo) * 100
            at_max = " ← МАКСИМУМ" if pct > 99 else ""
            at_min = " ← МИНИМУМ"  if pct < 1  else ""
            bounds_str = f"  [{lo:.4g} .. {hi:.4g}]  {pct:.0f}%{at_max}{at_min}"
        else:
            bounds_str = "  [фикс]"

        is_int_str = " (int)" if IS_INT[PARAM_NAMES.index(name)] else ""
        print(f"  │  {name:<22} = {val_str:<14}{is_int_str}{bounds_str}")
    
print(f"\n  ┌─ Адаптированные параметры SHADE")
print(f"  │  M_F  = {M_F.round(3)}")
print(f"  │  M_CR = {M_CR.round(3)}")
print(f"  │  → Оптимальный F ≈ {M_F.mean():.3f}, CR ≈ {M_CR.mean():.3f}")
