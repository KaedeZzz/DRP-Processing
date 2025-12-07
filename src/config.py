from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


@dataclass
class DRPConfig:
    """
    Unified DRP acquisition parameters and slice settings.
    """

    th_min: int
    th_max: int
    th_num: int
    ph_min: int
    ph_max: int
    ph_num: int
    phi_slice: int = 1
    theta_slice: int = 1

    def validate(self) -> None:
        if self.ph_min >= self.ph_max:
            raise ValueError("ph_min must be less than ph_max.")
        if self.th_min >= self.th_max:
            raise ValueError("th_min must be less than th_max.")
        if self.ph_num < 2 or self.th_num < 2:
            raise ValueError("ph_num and th_num must be at least 2.")
        if self.phi_slice < 1 or self.theta_slice < 1:
            raise ValueError("phi_slice and theta_slice must be positive.")

    @property
    def ph_step(self) -> float:
        return (self.ph_max - self.ph_min) / (self.ph_num - 1)

    @property
    def th_step(self) -> float:
        return (self.th_max - self.th_min) / (self.th_num - 1)

    def recompute_steps(self) -> None:
        """
        Included for compatibility; properties compute on the fly.
        """
        return


@dataclass
class CacheConfig:
    ph_slice: int = 1
    th_slice: int = 1


def load_drp_config(path: Path) -> DRPConfig:
    path = Path(path)
    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}

    required_keys = ["th_min", "th_max", "th_num", "ph_min", "ph_max", "ph_num"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in DRP config: {missing}")

    cfg = DRPConfig(
        th_min=data["th_min"],
        th_max=data["th_max"],
        th_num=data["th_num"],
        ph_min=data["ph_min"],
        ph_max=data["ph_max"],
        ph_num=data["ph_num"],
        phi_slice=data.get("phi_slice", 1),
        theta_slice=data.get("theta_slice", 1),
    )
    cfg.validate()
    return cfg


def load_cache_config(path: Path) -> CacheConfig:
    path = Path(path)
    if not path.exists():
        return CacheConfig()
    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    return CacheConfig(
        ph_slice=data.get("ph_slice", 1),
        th_slice=data.get("th_slice", 1),
    )


def save_cache_config(path: Path, cfg: CacheConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.dump(asdict(cfg), fh)
