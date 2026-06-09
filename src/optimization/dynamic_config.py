from dataclasses import dataclass

@dataclass
class DynamicMOEAConfig:
    dataset_path: str = "datasets/agrawal_abrupt.csv"

    # Ventana fija y NSGA-II
    window_size: int = 10
    pop_size: int = 20
    n_gen: int = 15
    seed: int = 42

    # Configuración inicial del Learn++NSE dinámico
    initial_a: float = 0.5
    initial_b: float = 5.0
    max_size: int = 20

    # Baseline fijo
    baseline_a: float = 0.5
    baseline_b: float = 5.0
    baseline_max_size: int = 20

    # Espacio de búsqueda del MOEA
    a_min: float = 0.1
    a_max: float = 2.0
    b_min: float = 1.0
    b_max: float = 15.0

    # Detalles técnicos y salida
    cache_decimals: int = 4
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

    if config.pop_size < 1:
        raise ValueError("pop_size debe ser >= 1.")

    if config.n_gen < 1:
        raise ValueError("n_gen debe ser >= 1.")

    if config.max_size < 1:
        raise ValueError("max_size debe ser >= 1.")

    if config.baseline_max_size < 1:
        raise ValueError("baseline_max_size debe ser >= 1.")

    if config.a_max < config.a_min:
        raise ValueError("Rango inválido para a.")

    if config.b_max < config.b_min:
        raise ValueError("Rango inválido para b.")