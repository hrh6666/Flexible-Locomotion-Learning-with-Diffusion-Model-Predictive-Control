#!/usr/bin/env python3
# postprocess_dataset.py
#
#   * Read <src>/<basename>.pt
#   * Replace obs/reward as needed
#   * Save to               <src>_new/<basename>_new.pt
#   * Copy config.json      <src>_new/config.json

import argparse
from pathlib import Path
import shutil
import numpy as np
import torch


def postprocess(src_dir: Path) -> None:
    """Process one trajectory folder and emit a new *_new folder."""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"{src_dir} is not a directory")

    # ----------------------------------------------------------------------
    # 1) locate files
    # ----------------------------------------------------------------------
    pt_files = list(src_dir.glob("*.pt"))
    if len(pt_files) != 1:
        raise RuntimeError(f"Expect exactly one .pt in {src_dir}, got {pt_files}")
    pt_in = pt_files[0]
    cfg_in = src_dir / "config.json"
    if not cfg_in.exists():
        raise FileNotFoundError(f"{cfg_in} not found")

    # derive new folder / filenames
    dst_dir = src_dir.with_name(src_dir.name + "_new")
    dst_dir.mkdir(parents=True, exist_ok=True)

    pt_out = dst_dir / (pt_in.stem + "_new" + pt_in.suffix)
    cfg_out = dst_dir / cfg_in.name  # same name

    # ----------------------------------------------------------------------
    # 2) load & modify
    # ----------------------------------------------------------------------
    trajectories = torch.load(pt_in)

    for traj in trajectories:
        obs_arr = np.asarray(traj["obs"], dtype=np.float32)  # (T, obs_dim)

        print("obs_arr.shape", obs_arr.shape)
        obs_arr[:, :3] = 0
        obs_arr[:, -1] = 0
        

        # overwrite
        traj["obs"] = obs_arr

    torch.save(trajectories, pt_out)
    print(f"[OK] processed .pt saved: {pt_out}")

    # ----------------------------------------------------------------------
    # 3) copy config.json verbatim
    # ----------------------------------------------------------------------
    shutil.copy2(cfg_in, cfg_out)
    print(f"[OK] copied config.json -> {cfg_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process a collected trajectory folder"
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Path to original folder: data/trajectory_YYYYMMDD_HHMMSS",
    )
    args = parser.parse_args()
    postprocess(args.src)


if __name__ == "__main__":
    main()