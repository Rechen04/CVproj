"""Finalize the sweep while the parent pipeline waits in SIGSTOP at the user's request."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from run_managed import ROOT, DEV, choose_setting, save, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--sweep", type=int, required=True)
    args = parser.parse_args()
    state_path = ROOT / "training/status.json"
    while True:
        stat_path = Path(f"/proc/{args.sweep}/stat")
        if not stat_path.exists():
            raise RuntimeError("Sweep process disappeared before its exit status was read")
        raw = stat_path.read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] == "Z":
            break
        time.sleep(10)
    state = json.loads(state_path.read_text())
    try:
        if int(fields[49]) != 0:
            raise RuntimeError(f"Sweep exited with wait status {fields[49]}; inspect training/sweep.log")
        expected = ["3dgs_3v"] + [f"dinit_conf_3v_wd{v}" for v in ["0.05", "0.1", "0.2"]]
        expected += [f"dinit_conf_3v_tau{v}" for v in ["0.05", "0.2"]]
        expected += [f"full_3v_wp{v}" for v in ["0.02", "0.05", "0.1"]] + ["full_3v_s10k"]
        for scene in DEV:
            for run in expected:
                validate(ROOT / "results_sweep" / scene / run, ply=False)
        params = choose_setting()
        state.update(status="paused", stage="sweep_complete", completed_stages=["smoke", "sweep"],
                     pause_reason="User requested a pause at the sweep milestone before main experiments",
                     frozen_parameters=params, resume_command=f"kill -CONT {args.parent}")
    except Exception as exc:
        state.update(status="paused_with_error", error=str(exc))
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(state_path, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
