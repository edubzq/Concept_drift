import copy
import time

import numpy as np

from src.optimization.dynamic_config import CandidateEvaluation
from src.utils.metrics import compute_ensemble_diversity, _safe_nanmean


def apply_learnpp_config(model, a, b, max_size, pruning_strategy, pruning_chunk=None):
    model.a = float(a)
    model.b = float(b)
    model.max_size = int(max_size)
    model.pruning_strategy = int(pruning_strategy)

    if pruning_chunk is None:
        model.prune_to_max_size()
    else:
        X, y = pruning_chunk
        model.prune_to_max_size(X=X, y=y)


def apply_candidate_config(model, candidate, pruning_chunk=None):
    apply_learnpp_config(
        model,
        a=candidate.a,
        b=candidate.b,
        max_size=candidate.max_size,
        pruning_strategy=candidate.pruning_strategy,
        pruning_chunk=pruning_chunk,
    )


def evaluate_learnpp_config_from_checkpoint(
    chunks,
    checkpoints,
    block_index,
    candidate,
    config,
):
    window_size = min(int(candidate.window_size), block_index + 1)
    window_start = block_index + 1 - window_size
    window_chunks = chunks[window_start:block_index + 1]

    model = copy.deepcopy(checkpoints[window_start])

    start = time.perf_counter()
    # No se pasa pruning_chunk aquí para evitar que la estrategia worst_accuracy
    # use etiquetas futuras de la ventana antes de evaluarla.
    apply_candidate_config(model, candidate, pruning_chunk=None)

    accuracies = []
    diversities = []
    ensemble_sizes = []

    for X, y in window_chunks:
        if model.models:
            preds, base_predictions = model.predict_with_base_predictions(X)
            accuracies.append(float(np.mean(preds == y)))
            diversities.append(
                compute_ensemble_diversity(model, base_predictions=base_predictions)
            )
            ensemble_sizes.append(len(model.models))

        model.fit_chunk(X, y)

    elapsed = time.perf_counter() - start

    recent_accuracy = _safe_nanmean(accuracies)
    diversity = _safe_nanmean(diversities)

    if config.use_elapsed_time_objective:
        complexity = float(elapsed)
    else:
        mean_size = _safe_nanmean(ensemble_sizes)
        if np.isnan(mean_size):
            mean_size = float(candidate.max_size)
        complexity = mean_size * max(window_size, 1)

    return CandidateEvaluation(
        recent_accuracy=recent_accuracy,
        diversity=diversity,
        complexity=float(complexity),
        elapsed=float(elapsed),
        window_size=int(window_size),
        pruning_strategy=int(candidate.pruning_strategy),
    )