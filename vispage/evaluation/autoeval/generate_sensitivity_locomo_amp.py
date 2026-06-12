"""Generate LoCoMO amp sensitivity configs.

This sweep reuses the main 8B LoCoMO scale4 semantic setup and varies only
page.max_amplification. Foreground update pages are allowed to register only up
to 1.2x the background amp limit.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "exp1_main" / "exp1_locomo_scale4_semantic_qwen3_embed_4b.json"
CONFIG_DIR = ROOT / "configs" / "sensitivity_locomo_amp"
CONFIG_LIST = ROOT / "evaluation" / "autoeval" / "configs_sensitivity_locomo_amp.txt"

AMPS = (2, 3, 7)
UPDATE_MULTIPLIER = 1.2


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base = read_json(BASE_CONFIG)
    paths: list[Path] = []
    for amp in AMPS:
        config = make_config(base, amp=amp)
        path = CONFIG_DIR / f"sensitivity_locomo_scale4_semantic_amp{amp}.json"
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        paths.append(path.relative_to(ROOT))
    CONFIG_LIST.write_text(
        "".join(f"{path.as_posix()}\n" for path in paths),
        encoding="utf-8",
    )
    print(f"wrote {len(paths)} configs to {CONFIG_DIR}")
    print(f"wrote {CONFIG_LIST}")


def make_config(base: dict[str, Any], *, amp: int) -> dict[str, Any]:
    config = deepcopy(base)
    config["run_name"] = "sensitivity_locomo_amp"
    config["page"]["key_prefix"] = f"sens-locomo-semantic-s4-amp{amp}"
    config["page"]["max_amplification"] = amp
    config["page"]["max_foreground_update_amplification"] = amp * UPDATE_MULTIPLIER
    config["renderer"]["output_dir"] = (
        f"paper_results/rendered_pages/sensitivity_locomo_amp/"
        f"locomo_scale4_semantic_amp{amp}"
    )
    config["metadata"] = {
        **dict(config.get("metadata", {})),
        "canonical_exp1": False,
        "sensitivity": "locomo_amp",
        "sensitivity_axis": "max_amplification",
        "sensitivity_value": amp,
        "foreground_update_amp_multiplier": UPDATE_MULTIPLIER,
        "foreground_update_max_amplification": amp * UPDATE_MULTIPLIER,
        "baseline_reference": (
            "paper_results/evaluation/exp1/autoeval/20260607_235821/"
            "config_004_exp1_locomo_scale4_baseline_qwen3_embed_4b"
        ),
        "amp5_reference": (
            "paper_results/evaluation/exp1/autoeval/20260607_235821/"
            "config_007_exp1_locomo_scale4_semantic_qwen3_embed_4b"
        ),
    }
    return config


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    main()
