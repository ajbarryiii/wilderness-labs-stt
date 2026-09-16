"""Serializable training circuit breakers; an accuracy plateau is not a collapse.

The loss reference can improve but cannot drift upward with a failing run. CTC
blank/empty alarms arm only after useful validation predictions have been seen.
Quantization phase transitions allow a short grace period, never for NaN/Inf.
"""

from __future__ import annotations

from collections import deque
from math import isfinite
from statistics import median
from typing import Any, Mapping


class TrainingCollapse(RuntimeError):
    def __init__(self, reason: str, metrics: Mapping[str, Any] | None = None):
        self.reason = reason
        self.metrics = dict(metrics or {})
        super().__init__(reason)


DEFAULTS = {
    "loss_window": 50,
    "min_train_steps": 100,
    "loss_explosion_factor": 5.0,
    "loss_explosion_patience": 3,
    "loss_reference_floor": 1e-3,
    "zero_grad_threshold": 1e-12,
    "zero_grad_patience": 50,
    "phase_grace_steps": 20,
    "val_patience": 3,
    "arm_wer": 0.8,
    "arm_empty_fraction": 0.2,
    "wer_deterioration_factor": 2.0,
    "wer_deterioration_absolute": 0.25,
    "empty_collapse_fraction": 0.9,
    "blank_collapse_fraction": 0.995,
}


class HealthMonitor:
    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = {**DEFAULTS, **dict(config or {})}
        for key in ("loss_window", "loss_explosion_patience", "zero_grad_patience", "val_patience"):
            if int(self.config[key]) < 1:
                raise ValueError(f"health.{key} must be at least 1")
        if self.config["loss_explosion_factor"] <= 1:
            raise ValueError("health.loss_explosion_factor must exceed 1")
        self.losses: deque[float] = deque(maxlen=int(self.config["loss_window"]))
        self.loss_reference: float | None = None
        self.loss_bad_count = 0
        self.zero_grad_count = 0
        self.validation_bad_count = 0
        self.validation_armed = False
        self.best_wer: float | None = None
        self.phase: str | None = None
        self.grace_until_step = -1
        self.last_step = -1

    @staticmethod
    def _finite(name: str, value: Any, metrics: Mapping[str, Any]) -> float:
        scalar = float(value)
        if not isfinite(scalar):
            raise TrainingCollapse(f"Nonfinite {name}", metrics)
        return scalar

    @staticmethod
    def _phase_name(quantization: Any) -> str | None:
        if quantization is None:
            return None
        if isinstance(quantization, str):
            return quantization
        if isinstance(quantization, Mapping):
            if "phase" in quantization:
                return str(quantization["phase"])
            if "weight" in quantization or "activation" in quantization:
                return "|".join(
                    f"{name}:{HealthMonitor._phase_name(quantization.get(name, 0))}"
                    for name in ("weight", "activation")
                )
            quantization = quantization.get("strength", quantization.get("progress"))
            if quantization is None:
                return None
        strength = float(quantization)
        if not isfinite(strength):
            raise TrainingCollapse("Nonfinite quantization strength", {"quantization": strength})
        return "full_precision" if strength <= 0 else "binary" if strength >= 1 else "ramp"

    def observe_train(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        quantization: Any = None,
    ) -> None:
        """Call once per optimizer step, with the pre-clipping gradient norm."""
        metrics = {"step": step, "loss": loss, "grad_norm": grad_norm}
        loss = self._finite("training loss", loss, metrics)
        grad_norm = self._finite("gradient norm", grad_norm, metrics)
        phase = self._phase_name(quantization)
        if phase is not None and phase != self.phase:
            if self.phase is not None:
                self.grace_until_step = step + int(self.config["phase_grace_steps"])
                self.loss_bad_count = self.zero_grad_count = self.validation_bad_count = 0
            self.phase = phase
        self.last_step = int(step)
        if step < self.grace_until_step:
            # Do not contaminate the healthy reference with transition losses.
            return

        ready = step >= int(self.config["min_train_steps"])
        if ready and self.loss_reference is not None:
            threshold = self.loss_reference * float(self.config["loss_explosion_factor"])
            if loss > threshold:
                self.loss_bad_count += 1
                if self.loss_bad_count >= int(self.config["loss_explosion_patience"]):
                    raise TrainingCollapse("Sustained training loss explosion", {
                        **metrics, "loss_reference": self.loss_reference,
                        "threshold": threshold, "consecutive_steps": self.loss_bad_count,
                    })
                # Exclude suspect observations rather than teaching the baseline
                # that a collapse is normal. A recovered observation resets this.
            else:
                self.loss_bad_count = 0
                self.losses.append(loss)
        else:
            self.losses.append(loss)

        if len(self.losses) == self.losses.maxlen:
            candidate = max(float(self.config["loss_reference_floor"]), median(self.losses))
            self.loss_reference = candidate if self.loss_reference is None else min(self.loss_reference, candidate)

        if ready and grad_norm <= float(self.config["zero_grad_threshold"]):
            self.zero_grad_count += 1
            if self.zero_grad_count >= int(self.config["zero_grad_patience"]):
                raise TrainingCollapse("Sustained zero gradient norm", {
                    **metrics, "consecutive_steps": self.zero_grad_count,
                })
        else:
            self.zero_grad_count = 0

    def observe_validation(self, step: int, metrics: Mapping[str, Any]) -> None:
        """WER/CER use fractions (0.2 means 20%); missing metrics are ignored."""
        checked = {"step": int(step), **dict(metrics)}
        values = {
            name: self._finite(f"validation {name}", metrics[name], checked)
            for name in ("wer", "cer", "loss", "blank_fraction", "empty_fraction")
            if metrics.get(name) is not None
        }
        if step < self.grace_until_step:
            return
        wer = values.get("wer")
        empty = values.get("empty_fraction")
        blank = values.get("blank_fraction")
        if not self.validation_armed:
            if (wer is not None and wer < float(self.config["arm_wer"])
                    and (empty is None or empty <= float(self.config["arm_empty_fraction"]))):
                self.validation_armed = True
                self.best_wer = wer
            return

        wer_bad = (
            wer is not None and self.best_wer is not None
            and wer > max(
                self.best_wer * float(self.config["wer_deterioration_factor"]),
                self.best_wer + float(self.config["wer_deterioration_absolute"]),
            )
        )
        empty_bad = empty is not None and empty >= float(self.config["empty_collapse_fraction"])
        # Blank-dominated CTC alignments can be healthy; require evidence that
        # decoded predictions also deteriorated before calling this a collapse.
        blank_bad = (
            blank is not None and blank >= float(self.config["blank_collapse_fraction"])
            and (wer_bad or (empty is not None and empty >= 0.5))
        )
        if wer_bad or empty_bad or blank_bad:
            self.validation_bad_count += 1
            if self.validation_bad_count >= int(self.config["val_patience"]):
                raise TrainingCollapse("Sustained validation collapse after learning", {
                    **checked, "best_wer": self.best_wer,
                    "consecutive_validations": self.validation_bad_count,
                    "wer_deteriorated": wer_bad, "empty_collapse": empty_bad,
                    "blank_collapse": blank_bad,
                })
        else:
            self.validation_bad_count = 0
            if wer is not None:
                self.best_wer = wer if self.best_wer is None else min(self.best_wer, wer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "losses": list(self.losses),
            **{name: getattr(self, name) for name in (
                "loss_reference", "loss_bad_count", "zero_grad_count",
                "validation_bad_count", "validation_armed", "best_wer",
                "phase", "grace_until_step", "last_step",
            )},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version", 1) != 1:
            raise ValueError("Unsupported health monitor state version")
        self.losses = deque((float(x) for x in state.get("losses", [])), maxlen=int(self.config["loss_window"]))
        for name in (
            "loss_reference", "loss_bad_count", "zero_grad_count",
            "validation_bad_count", "validation_armed", "best_wer",
            "phase", "grace_until_step", "last_step",
        ):
            if name in state:
                setattr(self, name, state[name])
