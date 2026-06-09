from dataclasses import dataclass


@dataclass(frozen=True)
class LearnPPCandidate:
    a: float
    b: float
    max_size: int
    window_size: int
    pruning_strategy: int


def _clip_int(value, lower, upper):
    rounded = int(round(value))
    return max(int(lower), min(rounded, int(upper)))


def decode_candidate(x, config, block_index=None):
    window_size = _clip_int(x[3], config.window_size_min, config.window_size_max)
    if block_index is not None:
        window_size = min(window_size, int(block_index) + 1)

    return LearnPPCandidate(
        a=float(x[0]),
        b=float(x[1]),
        max_size=_clip_int(x[2], config.max_size_min, config.max_size_max),
        window_size=window_size,
        pruning_strategy=_clip_int(
            x[4],
            config.pruning_strategy_min,
            config.pruning_strategy_max,
        ),
    )


def evaluation_cache_key(candidate, cache_decimals):
    return (
        round(float(candidate.a), cache_decimals),
        round(float(candidate.b), cache_decimals),
        int(candidate.max_size),
        int(candidate.window_size),
        int(candidate.pruning_strategy),
    )