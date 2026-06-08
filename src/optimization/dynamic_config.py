from dataclasses import dataclass


@dataclass
class DynamicMOEAConfig:
    dataset_path: str = "datasets/agrawal_abrupt.csv"

    # Ventana y reoptimización
    window_size: int = 10
    reopt_frequency: int = 5

    # Reoptimización event-driven
    use_event_reoptimization: bool = True
    accuracy_drop_threshold: float = 0.08
    accuracy_monitor_window: int = 3
    min_blocks_between_reopts: int = 5

    # NSGA-II
    pop_size: int = 20
    n_gen: int = 15
    seed: int = 42

    # Configuración inicial del ensemble dinámico
    initial_a: float = 0.5
    initial_b: float = 5.0
    initial_max_size: int = 20

    # Baseline fijo
    baseline_a: float = 0.5
    baseline_b: float = 5.0
    baseline_max_size: int = 20

    # Espacio de búsqueda
    a_min: float = 0.1
    a_max: float = 2.0
    b_min: float = 1.0
    b_max: float = 15.0
    max_size_min: int = 5
    max_size_max: int = 30

    # Detalles técnicos
    cache_decimals: int = 4
    use_elapsed_time_objective: bool = False

    # Salida
    output_dir: str = "results"
    plots_dir: str = "plots"
    verbose: bool = True


@dataclass
class CandidateEvaluation:
    recent_accuracy: float
    diversity: float
    complexity: float
    elapsed: float


def validate_dynamic_config(config: DynamicMOEAConfig):
    if config.window_size < 2:
        raise ValueError("window_size debe ser >= 2.")

    if config.reopt_frequency < 1:
        raise ValueError("reopt_frequency debe ser >= 1.")

    if config.accuracy_monitor_window < 2:
        raise ValueError("accuracy_monitor_window debe ser >= 2.")

    if config.accuracy_drop_threshold < 0:
        raise ValueError("accuracy_drop_threshold debe ser >= 0.")

    if config.min_blocks_between_reopts < 1:
        raise ValueError("min_blocks_between_reopts debe ser >= 1.")

    if config.pop_size < 1:
        raise ValueError("pop_size debe ser >= 1.")

    if config.n_gen < 1:
        raise ValueError("n_gen debe ser >= 1.")

    if config.max_size_min < 1 or config.max_size_max < config.max_size_min:
        raise ValueError("Rango inválido para max_size.")
