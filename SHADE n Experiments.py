"""
SHADE — 50 независимых экспериментов, сбор статистики → CSV.
"""

import sys
import time
import csv
import numpy as np

from shared_core import (
    ALL_PARAMS,
    evaluate_metrics_array, compute_objective,
)

# ─────────────────────────────────────────────────────────────────────────────
# Настройки
# ─────────────────────────────────────────────────────────────────────────────

N_EXP       = 30
SEEDS       = list(range(N_EXP))
CSV_PATH    = "shade_50_experiments.csv"

POP_SIZE    = 15
H           = 10
P_BEST      = 0.15
STALL_LIMIT = 150
MAXITER     = 3000
P_NOISE     = 0.15

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

print(f"\n── SHADE: {N_EXP} экспериментов ──")
print(f"pop={POP_SIZE}×{len(ALL_PARAMS)}={POP_SIZE*len(ALL_PARAMS)} | stall={STALL_LIMIT} | maxiter={MAXITER}")
print(f"Результаты → {CSV_PATH}\n")

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
# Один запуск SHADE
# ─────────────────────────────────────────────────────────────────────────────

def run_shade(seed):
    rng       = np.random.default_rng(seed)
    M_F       = np.full(H, 0.5)
    M_CR      = np.full(H, 0.5)
    k_mem     = 0
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
            F  = F_arr[i]
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
        nonlocal k_mem
        if not S_F: return
        w = np.array(weights); w = w / w.sum()
        f_arr = np.array(S_F); cr_arr = np.array(S_CR)
        M_F[k_mem]  = np.clip(np.sum(w*f_arr**2)/(np.sum(w*f_arr)+1e-12), 0.01, 1.0)
        M_CR[k_mem] = np.sum(w * cr_arr)
        k_mem = (k_mem + 1) % H

    # Запуск
    pop     = init_population()
    fitness = np.full(POP_SIZE, np.inf)
    for i in range(POP_SIZE):
        fitness[i] = objective(pop[i])
        n_calls += 1

    best_idx = np.argmin(fitness)
    gbest_f  = fitness[best_idx]
    gbest_x  = pop[best_idx].copy()

    stall = 0
    n_iter = 0
    for it in range(MAXITER):
        n_iter = it
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
        if stall >= STALL_LIMIT:
            converged = True
            break

    return {
        'gbest_f':   gbest_f,
        'gbest_x':   gbest_x,
        'n_calls':   n_calls,
        'n_iter':    n_iter,
        'converged': converged,
        'mean_F':    float(M_F.mean()),
        'mean_CR':   float(M_CR.mean()),
    }

# ─────────────────────────────────────────────────────────────────────────────
# CSV заголовок
# ─────────────────────────────────────────────────────────────────────────────

csv_header = (
    ["experiment", "seed", "elapsed_sec", "n_calls", "n_iter", "converged",
     "f_best", "performance_tops", "energy_eff", "energy_wt",
     "core_size", "ENOB", "Loss", "mean_F", "mean_CR"]
    + [f"param_{n}" for n in PARAM_NAMES]
)

# ─────────────────────────────────────────────────────────────────────────────
# Главный цикл
# ─────────────────────────────────────────────────────────────────────────────

total_start = time.time()

with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)

    for exp_idx, seed in enumerate(SEEDS):
        t0  = time.time()
        res = run_shade(seed)
        elapsed = time.time() - t0

        opt_x = res['gbest_x']
        try:
            energy, core_size, ENOB, perf, eff, Loss = evaluate_metrics_array(opt_x)
        except Exception:
            energy = core_size = ENOB = perf = eff = Loss = None

        row = (
            [exp_idx, seed, round(elapsed, 2), res['n_calls'], res['n_iter'],
             int(res['converged']), round(res['gbest_f'], 6),
             round(perf,      6) if perf      is not None else "",
             round(eff,       6) if eff       is not None else "",
             round(energy,    4) if energy    is not None else "",
             round(core_size, 1) if core_size is not None else "",
             round(ENOB,      4) if ENOB      is not None else "",
             round(Loss,      4) if Loss      is not None else "",
             round(res['mean_F'],  4),
             round(res['mean_CR'], 4)]
            + [round(float(v), 6) for v in opt_x]
        )
        writer.writerow(row)
        csvfile.flush()

        print(f"  [{exp_idx+1:2d}/{N_EXP}] seed={seed:3d} | "
              f"f={res['gbest_f']:.4e} | "
              f"perf={perf:.1f} TOPS | eff={eff:.2f} TOPS/W | "
              f"iter={res['n_iter']} | conv={'да' if res['converged'] else 'нет'} | "
              f"{elapsed:.1f}s")

total_elapsed = time.time() - total_start

# ─────────────────────────────────────────────────────────────────────────────
# Итоговая статистика
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  SHADE: {N_EXP} экспериментов за {total_elapsed/60:.1f} мин")
print(f"{'='*65}")

rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            rows.append({
                "f":    float(row["f_best"]),
                "perf": float(row["performance_tops"]) if row["performance_tops"] else None,
                "eff":  float(row["energy_eff"])       if row["energy_eff"]       else None,
                "E":    float(row["energy_wt"])         if row["energy_wt"]        else None,
                "N":    float(row["core_size"])         if row["core_size"]        else None,
                "conv": int(row["converged"]),
                "mF":   float(row["mean_F"]),
                "mCR":  float(row["mean_CR"]),
            })
        except Exception:
            pass

def stats(vals, label, fmt=".4e"):
    arr = np.array([v for v in vals if v is not None])
    if not len(arr): return
    cv = abs(np.std(arr)/np.mean(arr))*100 if np.mean(arr) != 0 else 0
    print(f"  {label}:")
    print(f"    min={np.min(arr):{fmt}} | median={np.median(arr):{fmt}} | "
          f"mean={np.mean(arr):{fmt}} | std={np.std(arr):{fmt}} | CV={cv:.1f}%")

conv_n = sum(r["conv"] for r in rows)
print(f"\n  Сошлись: {conv_n}/{N_EXP} ({conv_n/N_EXP*100:.0f}%)")
stats([r["f"]    for r in rows], "f(x)")
stats([r["perf"] for r in rows], "Performance TOPS", fmt=".2f")
stats([r["eff"]  for r in rows], "Energy eff TOPS/W", fmt=".4f")
stats([r["E"]    for r in rows], "Energy Вт",          fmt=".2f")

mF_arr  = np.array([r["mF"]  for r in rows])
mCR_arr = np.array([r["mCR"] for r in rows])
print(f"\n  Адаптированные параметры:")
print(f"    M_F  медиана = {np.median(mF_arr):.3f}")
print(f"    M_CR медиана = {np.median(mCR_arr):.3f}")

best = min(rows, key=lambda r: r["f"])
print(f"\n  Лучший: seed={SEEDS[rows.index(best)]}")
print(f"    f={best['f']:.6e} | perf={best['perf']:.2f} TOPS | "
      f"eff={best['eff']:.4f} TOPS/W | E={best['E']:.1f} Вт")
print(f"\n  → {CSV_PATH}")
