import time
import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from learnpp_nse import LearnPPNSE
from test_ensembles import load_blocks


# ============================================
# Configuración
# ============================================

DATASET_PATH = "datasets/agrawal_abrupt.csv"

# Puedes usar solo una parte reciente si quieres acelerar pruebas
USE_RECENT_WINDOW = False
RECENT_N_BLOCKS = 10

# Rango de búsqueda
A_MIN, A_MAX = 0.1, 2.0
B_MIN, B_MAX = 1.0, 15.0
MAX_SIZE_MIN, MAX_SIZE_MAX = 5, 30

# NSGA-II
POP_SIZE = 20
N_GEN = 15
SEED = 42


# ============================================
# Diversidad del ensemble
# ============================================

def compute_ensemble_diversity(model, X):
    """
    Diversidad media por pares basada en disagreement.
    """
    if not hasattr(model, "models") or len(model.models) < 2:
        return 0.0

    X_dict = X.to_dict(orient="records")
    preds = []

    for base_model in model.models:
        model_preds = []
        for xi in X_dict:
            pred = base_model.predict_one(xi)
            model_preds.append(pred)
        preds.append(model_preds)

    preds = np.array(preds)
    n_models = preds.shape[0]

    total_disagreement = 0.0
    total_pairs = 0

    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagreement = np.mean(preds[i] != preds[j])
            total_disagreement += disagreement
            total_pairs += 1

    if total_pairs == 0:
        return 0.0

    return float(total_disagreement / total_pairs)


# ============================================
# Evaluación de una configuración
# ============================================

def evaluate_learnpp_config(chunks, a, b, max_size):
    """
    Evalúa una configuración concreta de LearnPPNSE.
    Devuelve:
        mean_acc
        mean_div
        elapsed_time
    """
    model = LearnPPNSE(
        a=float(a),
        b=float(b),
        max_size=int(round(max_size))
    )

    accuracies = []
    diversities = []

    start = time.perf_counter()

    for i, (X, y) in enumerate(chunks):
        if i > 0:
            preds = np.array(model.predict(X))
            acc = np.mean(preds == y)
            accuracies.append(acc)

            div = compute_ensemble_diversity(model, X)
            diversities.append(div)

        model.fit_chunk(X, y)

    elapsed = time.perf_counter() - start

    mean_acc = float(np.mean(accuracies)) if len(accuracies) > 0 else 0.0
    mean_div = float(np.mean(diversities)) if len(diversities) > 0 else 0.0

    return mean_acc, mean_div, elapsed


# ============================================
# Problema multiobjetivo
# ============================================

class LearnPPNSGAProblem(ElementwiseProblem):
    def __init__(self, chunks):
        self.chunks = chunks

        super().__init__(
            n_var=3,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array([A_MIN, B_MIN, MAX_SIZE_MIN], dtype=float),
            xu=np.array([A_MAX, B_MAX, MAX_SIZE_MAX], dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        a = float(x[0])
        b = float(x[1])
        max_size = int(round(x[2]))

        acc, div, elapsed = evaluate_learnpp_config(
            self.chunks,
            a=a,
            b=b,
            max_size=max_size
        )

        # pymoo minimiza
        f1 = -acc
        f2 = -div
        f3 = elapsed

        out["F"] = np.array([f1, f2, f3], dtype=float)


# ============================================
# Selección de una solución final del frente
# ============================================

def choose_compromise_solution(res):
    """
    Selección simple de una solución de compromiso.
    score = 0.6*acc + 0.3*div - 0.1*cost_normalized
    """
    F = np.array(res.F, dtype=float)
    X = np.array(res.X, dtype=float)

    acc = -F[:, 0]
    div = -F[:, 1]
    cost = F[:, 2]

    # normalización simple
    cost_min, cost_max = np.min(cost), np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)

    score = 0.6 * acc + 0.3 * div - 0.1 * cost_norm
    best_idx = int(np.argmax(score))

    return best_idx, X[best_idx], F[best_idx], score[best_idx]


# ============================================
# Utilidad para mostrar resultados
# ============================================

def pareto_to_df(res):
    rows = []

    for x, f in zip(res.X, res.F):
        a = float(x[0])
        b = float(x[1])
        max_size = int(round(x[2]))

        acc = -float(f[0])
        div = -float(f[1])
        cost = float(f[2])

        rows.append({
            "a": round(a, 4),
            "b": round(b, 4),
            "max_size": max_size,
            "accuracy": round(acc, 6),
            "diversity": round(div, 6),
            "time": round(cost, 6),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(by=["accuracy", "diversity"], ascending=[False, False]).reset_index(drop=True)


# ============================================
# Main
# ============================================

def main():
    chunks = load_blocks(DATASET_PATH)

    if USE_RECENT_WINDOW:
        chunks = chunks[-RECENT_N_BLOCKS:]

    print(f"Dataset: {DATASET_PATH}")
    print(f"Número de bloques usados: {len(chunks)}")

    problem = LearnPPNSGAProblem(chunks)

    algorithm = NSGA2(
        pop_size=POP_SIZE
    )

    termination = get_termination("n_gen", N_GEN)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=SEED,
        verbose=True
    )

    df = pareto_to_df(res)
    print("\n=== Frente de Pareto encontrado ===")
    print(df)

    best_idx, best_x, best_f, best_score = choose_compromise_solution(res)

    best_a = float(best_x[0])
    best_b = float(best_x[1])
    best_max_size = int(round(best_x[2]))

    best_acc = -float(best_f[0])
    best_div = -float(best_f[1])
    best_time = float(best_f[2])

    print("\n=== Solución de compromiso elegida ===")
    print(f"a = {best_a:.4f}")
    print(f"b = {best_b:.4f}")
    print(f"max_size = {best_max_size}")
    print(f"accuracy = {best_acc:.6f}")
    print(f"diversity = {best_div:.6f}")
    print(f"time = {best_time:.6f}")
    print(f"score = {best_score:.6f}")

    df.to_csv("pareto_learnpp.csv", index=False)
    print("\nFrente guardado en: pareto_learnpp.csv")


if __name__ == "__main__":
    main()