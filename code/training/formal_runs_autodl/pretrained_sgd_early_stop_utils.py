from __future__ import annotations

import math
from typing import Any


def parse_epoch(value: Any) -> int | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def devices_equivalent(expected: Any, observed: Any) -> bool:
    """Accept serialization-only differences such as integer 0 versus string '0'."""

    if isinstance(expected, (list, tuple)) or isinstance(observed, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or not isinstance(observed, (list, tuple)):
            return False
        return [str(item).strip() for item in expected] == [str(item).strip() for item in observed]
    return str(expected).strip() == str(observed).strip()


def is_duplicate_result_epoch(result_epoch: Any, recorded_epoch: Any) -> bool:
    result = parse_epoch(result_epoch)
    recorded = parse_epoch(recorded_epoch)
    return result is not None and recorded is not None and result <= recorded


def proves_locked_early_stop(
    *,
    final_epoch: Any,
    best_epoch: Any,
    target_epochs: int,
    patience: int,
    history_contiguous: bool,
    history_finite: bool,
    weights_present: bool,
) -> bool:
    final_value = parse_epoch(final_epoch)
    best_value = parse_epoch(best_epoch)
    return bool(
        history_contiguous
        and history_finite
        and weights_present
        and final_value is not None
        and best_value is not None
        and final_value < int(target_epochs)
        and final_value - best_value >= int(patience)
    )


def classify_normal_training_return(
    *,
    final_epoch: Any,
    target_epochs: int,
    trainer_epoch_zero_based: Any,
    stopper_best_epoch_one_based: Any,
    patience: int,
) -> str:
    """Classify an Ultralytics train return using its actual epoch conventions.

    Ultralytics passes ``epoch + 1`` into ``EarlyStopping``. Consequently,
    ``stopper.best_epoch`` is already one-based, while ``trainer.epoch`` is
    zero-based. Keeping that distinction explicit avoids the historical
    off-by-one completion failure.
    """

    final_value = parse_epoch(final_epoch)
    trainer_value = parse_epoch(trainer_epoch_zero_based)
    stopper_best = parse_epoch(stopper_best_epoch_one_based)
    if final_value is None or trainer_value is None or final_value != trainer_value + 1:
        return "normal_return_incomplete"
    if final_value >= int(target_epochs):
        return "max_epochs"
    if stopper_best is None or stopper_best <= 0:
        return "normal_return_incomplete"
    if final_value - stopper_best >= int(patience):
        return "early_stopping"
    return "normal_return_incomplete"
