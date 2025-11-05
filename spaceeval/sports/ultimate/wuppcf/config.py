from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

PROVIDER_ULTIMATE_TRACK: str = "UltimateTrack"
PROVIDER_UFA: str = "UFA"


def _load_all_params() -> Dict[str, Dict[str, float]]:
    """Load all parameters from params.yaml"""
    config_path = Path(__file__).parent / "params.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_ALL_PARAMS = _load_all_params()


@dataclass(frozen=True)
class ProviderSettings:
    """Provider-specific settings"""

    name: str
    field_length: float
    field_width: float
    fps: int

    @property
    def field_dimen(self) -> Tuple[float, float]:
        """Return field dimensions as (length, width) tuple"""
        return self.field_length, self.field_width


def get_provider_settings(provider: str) -> ProviderSettings:
    """Get provider settings from YAML config.

    Args:
        provider: Provider name (e.g., "UltimateTrack", "UFA")

    Returns:
        ProviderSettings object

    Example:
        >>> settings = get_provider_settings("UltimateTrack")
        >>> settings.field_length  # 94.0
    """
    if provider not in _ALL_PARAMS:
        available = ", ".join(sorted([p for p in _ALL_PARAMS.keys()]))
        raise ValueError(
            f"Unknown provider '{provider}'. Available providers: {available}"
        )

    config = _ALL_PARAMS[provider]
    return ProviderSettings(
        name=provider,
        field_length=config["field_length"],
        field_width=config["field_width"],
        fps=config["fps"],
    )


def get_model_params(
    provider: str, time_to_control_veto: int = None
) -> Dict[str, float]:
    """Get model parameters from YAML config."""
    if provider not in _ALL_PARAMS or provider == "model_params":
        available = ", ".join(
            sorted([p for p in _ALL_PARAMS.keys() if p != "model_params"])
        )
        raise ValueError(
            f"Unknown provider '{provider}'. Available providers: {available}"
        )

    params = _ALL_PARAMS["model_params"].copy()

    params["max_player_accel"] /= params["grid_size"]
    params["max_player_speed"] /= params["grid_size"]
    params["average_ball_speed"] /= params["grid_size"]

    settings = get_provider_settings(provider)
    params["int_dt"] = 1.0 / settings.fps

    if time_to_control_veto is not None:
        params["time_to_control_veto"] = time_to_control_veto

    veto = params["time_to_control_veto"]
    params["time_to_control_att"] = (
        veto
        * np.log(10)
        * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        veto
        * np.log(10)
        * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_def"])
    )

    return params
