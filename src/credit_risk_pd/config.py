from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Paths:
    root: Path
    data_processed: Path
    data_raw: Path
    outputs: Path

def get_paths() -> Paths:
    root_env = os.getenv("CRPD_ROOT", "").strip()
    raw_env = os.getenv("CRPD_DATA_RAW", "").strip()
    proc_env = os.getenv("CRPD_DATA_PROCESSED", "").strip()
    out_env = os.getenv("CRPD_OUTPUTS", "").strip()

    repo_root = Path(__file__).resolve().parents[3]

    root = Path(root_env).expanduser().resolve() if root_env else repo_root
    data_raw = Path(raw_env).expanduser().resolve() if raw_env else (root / "data")
    data_processed = Path(proc_env).expanduser().resolve() if proc_env else (root / "data" / "processed")
    outputs = Path(out_env).expanduser().resolve() if out_env else (root / "outputs")

    outputs.mkdir(parents=True, exist_ok=True)

    return Paths(
        root=root,
        data_processed=data_processed,
        data_raw=data_raw,
        outputs=outputs,
    )

