"""Observe validation without changing predictions, loss values, RNG, or gradients."""
import math
import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


class ValidationAudit:
    def __init__(self):
        self.trainer = None
        self.handles = []
        self.record = None

    def start(self, validator):
        if not validator.training:
            return
        assert not self.handles
        trainer = self.trainer
        net = trainer.ema.ema if trainer.ema else trainer.model
        # The historical loss creates the same criterion lazily; this permits a read-only hook.
        if getattr(net, "criterion", None) is None:
            net.criterion = net.init_criterion()
        self.record = {
            "epoch": int(trainer.epoch) + 1,
            "expected_batches": len(validator.dataloader),
            "prediction_batches": 0, "bce_batches": 0, "overflow_batches": 0,
            "max_bce_sum_fp32": 0.0, "batches": [],
            "predictions_and_elementwise_bce_finite": True,
        }

        def predictions(module, inputs, output):
            values = list(tensors(output))
            if not values or not all(bool(torch.isfinite(t).all()) for t in values):
                raise FloatingPointError("Non-finite validation prediction")
            self.record["prediction_batches"] += 1

        def bce(module, inputs, output):
            if not all(bool(torch.isfinite(t).all()) for t in [*inputs, output]):
                raise FloatingPointError("Non-finite validation BCE input or element")
            original_sum = float(output.sum())
            wide_sum = float(output.float().sum())
            if not math.isfinite(wide_sum):
                raise FloatingPointError("Non-finite FP32 validation BCE reduction")
            overflow = original_sum == math.inf
            if not math.isfinite(original_sum) and not (overflow and output.dtype == torch.float16):
                raise FloatingPointError("Unexplained non-finite validation BCE reduction")
            self.record["bce_batches"] += 1
            self.record["overflow_batches"] += int(overflow)
            self.record["max_bce_sum_fp32"] = max(self.record["max_bce_sum_fp32"], wide_sum)
            self.record["batches"].append({
                "index": self.record["bce_batches"] - 1,
                "elements": output.numel(), "dtype": str(output.dtype),
                "sum_original": "+inf" if overflow else original_sum,
                "sum_fp32_diagnostic": wide_sum,
            })
            # No returned tensor: PyTorch retains the unmodified historical BCE output.

        self.handles = [net.register_forward_hook(predictions),
                        net.criterion.bce.register_forward_hook(bce)]

    def end(self, validator):
        if not validator.training:
            return
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def check(self, trainer, metrics):
        record = self.record
        assert record is not None and record["epoch"] == int(trainer.epoch) + 1
        assert not self.handles
        assert record["prediction_batches"] == record["bce_batches"] == record["expected_batches"]
        bad = {key: value for key, value in metrics.items() if not math.isfinite(value)}
        if bad:
            if set(bad) != {"val/cls_loss"} or bad["val/cls_loss"] != math.inf:
                raise FloatingPointError("Non-finite validation metric outside proven FP16 BCE overflow")
            if not record["overflow_batches"]:
                raise FloatingPointError("Validation inf has no observed finite-element FP16 overflow")
            for name, value in [
                ("training model", trainer.model.state_dict()),
                ("EMA model", trainer.ema.ema.state_dict()),
                ("optimizer", trainer.optimizer.state_dict()),
            ]:
                if not all(bool(torch.isfinite(t).all()) for t in tensors(value)):
                    raise FloatingPointError("Non-finite " + name)
        elif record["overflow_batches"]:
            raise FloatingPointError("Observed reduction overflow not reflected in validation loss")
        return {
            "metrics": {key: value if math.isfinite(value) else None for key, value in metrics.items()},
            "raw_nonfinite_metrics": {key: "+inf" for key in bad},
            "validation_loss_audit": record,
        }
