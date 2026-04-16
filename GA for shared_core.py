import sys
import time
import numpy as np

try:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.mixed import (
        MixedVariableMating,
        MixedVariableSampling,
        MixedVariableDuplicateElimination,
    )
    from pymoo.core.problem import Problem
    from pymoo.core.variable import Real, Integer
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize as pymoo_minimize
except ImportError:
    print("pip install pymoo")
    sys.exit(1)

from shared_core import (
    ALL_PARAMS,
    evaluate_metrics_dict, compute_objective,
)

# ─────────────────────────────────────────────────────────────────────────────
# Настройки
# ─────────────────────────────────────────────────────────────────────────────

SEED       = 42
POP_SIZE   = 100
STALL_GENS = 150     # эквивалентно SHADE: 50 × 100 = 5000 вызовов без улучшения
MAX_GENS   = 1500    # жёсткая остановка
PRINT_EACH = 10

INT_PARAMS = {"cores", "num_soa", "num_matrices (n)", "batch_size (b)", "vector_size (v)"}

PARAM_NAMES   = [name for name, *_ in ALL_PARAMS]
IS_INT        = np.array([name in INT_PARAMS for name in PARAM_NAMES], dtype=bool)
LB            = np.array([lo for _, _, lo, hi in ALL_PARAMS], dtype=float)
UB            = np.array([hi for _, _, lo, hi in ALL_PARAMS], dtype=float)
FIXED_VALUES  = {name: init for name, init, lo, hi in ALL_PARAMS if lo == hi}
ACTIVE_PARAMS = [(name, init, lo, hi) for name, init, lo, hi in ALL_PARAMS if lo != hi]

GROUPS = {
    "Параметры задачи": ["num_matrices (n)", "batch_size (b)", "vector_size (v)"],
    "Параметры чипа":   ["cores", "freq_mhz", "freq_mat_mhz", "bw_gbps", "buffer_size_mb"],
    "Оптические":       ["n_wpe", "E_elop", "E_elop_driv_vec", "E_elop_driv_mat",
                         "P_meminterf", "P_mat_to", "E_afe", "E_adc_fom",
                         "n_soa", "num_soa", "IL_splitter", "IL_FtoC", "IL_SMF",
                         "IL_WG", "L_MZI_TO", "L_MZI_EL", "IL_DC", "IL_penalty",
                         "IL_to_ps", "IL_el_ps", "IL_Crossing", "RIN", "I_d",
                         "R_PD", "R_l"],
}

VARS = {}
for name, init, lo, hi in ACTIVE_PARAMS:
    if name in INT_PARAMS:
        VARS[name] = Integer(bounds=(int(lo), int(hi)))
    else:
        VARS[name] = Real(bounds=(float(lo), float(hi)))

print(f"GA | pop={POP_SIZE} | stall={STALL_GENS} | max_gen={MAX_GENS} | seed={SEED} | запускаю...")

# ─────────────────────────────────────────────────────────────────────────────
# Задача
# ─────────────────────────────────────────────────────────────────────────────

class OPUProblem(Problem):
    def __init__(self, call_count, best_so_far):
        super().__init__(vars=VARS, n_obj=1)
        self._calls = call_count
        self._best  = best_so_far

    def _evaluate(self, X, out, *args, **kwargs):
        fitnesses = []
        for x_dict in X:
            params = {**FIXED_VALUES, **x_dict}
            try:
                energy, core_size, ENOB, perf, eff, Loss = evaluate_metrics_dict(params)
                val, ok = compute_objective(energy, core_size, ENOB, perf, eff, Loss)
            except Exception:
                val, ok = None, False
            result = val if ok else 1e10

            '''
            if self._calls[0] <= 5:
                print(f"  #{self._calls[0]}: ok={ok}, val={val}, "
                      f"E={energy if energy else 'None'}, "
                      f"ENOB={ENOB if ENOB else 'None'}, "
                      f"N={core_size if core_size else 'None'}")
            '''
            
            if result < self._best[0]:
                #rel = abs(result - self._best[0]) / (abs(self._best[0]) + 1e-30)
                #if rel > 1e-6:
                self._best[0] = result
            fitnesses.append(result)
        out["F"] = [[f] for f in fitnesses]

# ─────────────────────────────────────────────────────────────────────────────
# Callback — логирует прогресс и считает stall по глобальному лучшему
# ─────────────────────────────────────────────────────────────────────────────

class StallCallback(Callback):
    def __init__(self, stall_limit, call_count, best_so_far, t0):
        super().__init__()
        self.stall_limit = stall_limit
        self.call_count  = call_count
        self.best_so_far = best_so_far
        self.t0          = t0
        self.gen         = 0
        self.stall       = 0
        self.prev_best   = np.inf
        self.converged   = False

    def notify(self, algorithm):
        self.gen += 1
        current = self.best_so_far[0]

        if self.prev_best == np.inf:
            rel = 1.0
        else:
            rel = abs(current - self.prev_best) / (abs(self.prev_best) + 1e-30)

        if current < self.prev_best and rel > 1e-6:
            self.stall     = 0
            self.prev_best = current
        else:
            self.stall += 1

        if PRINT_EACH and (self.gen % PRINT_EACH == 0 or self.gen == 1):
            elapsed = time.time() - self.t0
            print(f"  gen {self.gen:4d} | calls {self.call_count[0]:7d} | "
                  f"best {self.best_so_far[0]:.4e} | "
                  f"stall {self.stall:3d}/{self.stall_limit} | "
                  f"{elapsed:.1f}s")

        if self.stall >= self.stall_limit:
            self.converged = True
            algorithm.termination.force_termination = True
            print(f"  [STALL] gen {self.gen} | best {self.best_so_far[0]:.4e} — достигнут stall_limit")

# ─────────────────────────────────────────────────────────────────────────────
# Запуск
# ─────────────────────────────────────────────────────────────────────────────

call_count  = [0]
best_so_far = [np.inf]
t0          = time.time()

problem  = OPUProblem(call_count, best_so_far)
callback = StallCallback(STALL_GENS, call_count, best_so_far, t0)

algorithm = GA(
    pop_size = POP_SIZE,
    sampling = MixedVariableSampling(),
    mating   = MixedVariableMating(
        eliminate_duplicates=MixedVariableDuplicateElimination()
    ),
    eliminate_duplicates=MixedVariableDuplicateElimination(),
)

result = pymoo_minimize(
    problem, algorithm,
    ("n_gen", MAX_GENS),
    seed=SEED, callback=callback, verbose=False,
)

elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# Итог
# ─────────────────────────────────────────────────────────────────────────────

best_params = {**FIXED_VALUES, **result.X}
try:
    energy, core_size, ENOB, perf, eff, Loss = evaluate_metrics_dict(best_params)
except Exception:
    energy = core_size = ENOB = perf = eff = Loss = None

print(f"\n{'='*65}")
print(f"  {'Сошёлся' if callback.converged else 'Бюджет исчерпан'} | "
      f"поколение {callback.gen} | {elapsed:.1f}s | вызовов {call_count[0]:,}")
print(f"{'='*65}")
print(f"  f(x)          = {best_so_far[0]:.9e}")
print(f"  Performance   = {perf:.2f} TOPS"  if perf      else "  Performance  = —")
print(f"  Energy        = {energy:.2f} Вт"  if energy    else "  Energy       = —")
print(f"  Core size N   = {core_size:.1f}"  if core_size else "  Core size    = —")
print(f"  ENOB          = {ENOB:.4f}"        if ENOB      else "  ENOB         = —")
print(f"  Energy eff    = {eff:.4f} TOPS/W" if eff       else "  Energy eff   = —")
print(f"  Loss          = {Loss:.4f}"        if Loss      else "  Loss         = —")

print(f"\n{'─'*65}")
print(f"  ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ")
print(f"{'─'*65}")

for group_name, param_list in GROUPS.items():
    print(f"\n  +-- {group_name}")
    for name in param_list:
        if name not in best_params:
            continue
        val = float(best_params[name])
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
            pct        = (val - lo) / (hi - lo) * 100
            at_max     = " <- МАКСИМУМ" if pct > 99 else ""
            at_min     = " <- МИНИМУМ"  if pct < 1  else ""
            bounds_str = f"  [{lo:.4g} .. {hi:.4g}]  {pct:.0f}%{at_max}{at_min}"
        else:
            bounds_str = "  [фикс]"

        is_int_str = " (int)" if IS_INT[idx] else ""
        print(f"  |  {name:<22} = {val_str:<14}{is_int_str}{bounds_str}")
