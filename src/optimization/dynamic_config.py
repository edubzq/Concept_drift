from dataclasses import dataclass


@dataclass
class DynamicMOEAConfig:
    dataset_path: str = "datasets/agrawal_abrupt.csv"
    window_size: int = 10
    reopt_frequency: int = 5
    pop_size: int = 20
    n_gen: int = 15
    seed: int = 42
    initial_a: float = 0.5
    initial_b: float = 5.0
    initial_max_size: int = 20
    baseline_a: float = 0.5
    baseline_b: float = 5.0
    baseline_max_size: int = 20
    a_min: float = 0.1
    a_max: float = 2.0
    b_min: float = 1.0
    b_max: float = 15.0
    max_size_min: int = 5
    max_size_max: int = 30
    drop_threshold: float = 0.15
    pre_window: int = 3
    recovery_ratio: float = 0.60
    local_min_window: int = 5
    cache_decimals: int = 4
    use_elapsed_time_objective: bool = False
    output_dir: str = "results"
    plots_dir: str = "plots"
    verbose: bool = True


@dataclass
class CandidateEvaluation:
    accuracy: float
    accuracy_min: float
    diversity: float
    recovery_time: float
    cost: float
    elapsed: float


def validate_dynamic_config(config):
    if config.window_size < 2:
        raise ValueError("window_size debe ser >= 2 para evaluar accuracy en la ventana.")
    if config.reopt_frequency < 1:
        raise ValueError("reopt_frequency debe ser >= 1.")
    if config.pop_size < 1:
        raise ValueError("pop_size debe ser >= 1.")
    if config.n_gen < 1:
        raise ValueError("n_gen debe ser >= 1.")
    if config.max_size_min < 1 or config.max_size_max < config.max_size_min:
        raise ValueError("Rango inválido para max_size.")
    if config.drop_threshold < 0:
        raise ValueError("drop_threshold debe ser >= 0.")
    if not 0 < config.recovery_ratio <= 1:
        raise ValueError("recovery_ratio debe estar en (0, 1].")