#!/usr/bin/env python3
"""
Automated SNR / noise-type sweep for the ecHT phase-estimation graph.

For every (snr_db, noise_color) combination on the grid this script:
  1. patches ``SimulatedSource.options.snr_db`` / ``noise_color`` in the graph config
     (writing a throwaway ``_sweep_<graph>`` copy so the original is untouched),
  2. runs one Falcon session head-less -- no keypress handling, no stim
     protocol -- and waits for Falcon to auto-exit after ``n_messages``,
  3. moves that run's ``results/<run>`` folder into a single sweep directory.

When every run is done it builds the phase-error-vs-SNR figure by calling
``analysis.main.analyse_snr_sweep`` on the sweep directory.

Examples
--------
    python sweep_clas.py
    python sweep_clas.py --snr -10 0 10 20 50 --noise white pink --name my_sweep
    python sweep_clas.py --no-analyse          # just collect the runs
    python sweep_clas.py --dry-run             # print the plan and exit
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
import zmq

REPO_ROOT     = Path(__file__).resolve().parent
FALCON_BIN    = REPO_ROOT / "build" / "release" / "falcon" / "falcon"
FALCON_CONFIG = REPO_ROOT / ".falcon" / "config.yaml"
RESULTS_DIR   = "results"

sys.path.insert(0, str(REPO_ROOT))


def terminate(proc: subprocess.Popen) -> None:
    """SIGTERM (then SIGKILL) the Falcon process group. Each run is started
    with start_new_session=True, so the process group is the child's alone
    and this never touches the sweep script itself."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def patch_graph(base_graph_path: Path, out_graph_path: Path,
                snr_db: float, noise_color: str, n_messages: int | None) -> float:
    """Write a copy of the graph config with the SimulatedSource's SNR / noise
    colour (and optionally n_messages) overridden. Returns the length of the
    generated signal in seconds (SimulatedSource n_messages * nsamples / fs)."""
    with open(base_graph_path) as f:
        cfg = yaml.safe_load(f)

    producer = cfg["graph"]["processors"]["SimulatedSource"]
    opts = producer["options"]
    opts["snr_db"]      = float(snr_db)
    opts["noise_color"] = noise_color

    if n_messages is not None:
        # The &n_messages anchor is already expanded by safe_load, so every
        # processor that referenced it carries its own copy -- patch each.
        for proc_cfg in cfg["graph"]["processors"].values():
            if isinstance(proc_cfg, dict) and "n_messages" in proc_cfg.get("options", {}):
                proc_cfg["options"]["n_messages"] = int(n_messages)

    with open(out_graph_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    fs        = cfg["graph"].get("defaults", {}).get("fs") or opts.get("fs", 10000)
    nsamples  = opts.get("nsamples", 1)
    return opts["n_messages"] * nsamples / float(fs)


def set_falcon_log_path(log_path: Path) -> None:
    with open(FALCON_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("logging", {})["path"] = str(log_path)
    with open(FALCON_CONFIG, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_one(ctx: zmq.Context, port: int, graph_name: str, out_graph_path: Path,
            snr_db: float, noise_color: str, sweep_dir: Path,
            run_seconds: float, shutdown_s: float) -> Path | None:
    """Launch one Falcon run for the already-patched graph at
    ``out_graph_path``, let it stream for ``run_seconds``, then stop and quit
    the graph (the scripted equivalent of pressing 's' then 'q' in clas.py)
    and move its results folder into ``sweep_dir``. Returns the destination
    path, or None if the run failed."""
    run_name        = f"snr{snr_db:g}_{noise_color}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_results_dir = Path(RESULTS_DIR) / run_name
    run_results_dir.mkdir(parents=True, exist_ok=True)
    set_falcon_log_path(run_results_dir / f"{run_name}.log")

    print(f"\n=== SNR {snr_db:g} dB, {noise_color} noise  ->  {run_results_dir} ===")

    proc = subprocess.Popen(
        [str(FALCON_BIN), out_graph_path.name, "--config", str(FALCON_CONFIG)],
        stdin=subprocess.PIPE, start_new_session=True,
    )
    time.sleep(1.0)

    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 15000)
    socket.connect(f"tcp://127.0.0.1:{port}")

    ok = False
    try:
        socket.send_multipart([b"graph", b"start", RESULTS_DIR.encode(), run_name.encode(), b""])
        socket.recv_multipart()

        # This graph doesn't self-terminate: wait for the fixed signal length
        # (plus margin) while Falcon streams, checking it hasn't died early.
        print(f"  streaming for {run_seconds:g}s ...")
        deadline = time.monotonic() + run_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print("  Falcon exited early; aborting this run.")
                return None
            time.sleep(0.5)

        # Scripted 's' then 'q'.
        socket.send_multipart([b"graph", b"stop"]); socket.recv_multipart()
        socket.send_multipart([b"quit"]);           socket.recv_multipart()
        try:
            proc.wait(timeout=shutdown_s)
            ok = True
        except subprocess.TimeoutExpired:
            print(f"  Falcon did not quit within {shutdown_s:g}s; terminating.")
            ok = True  # data is already written; shutdown was just slow
    except zmq.ZMQError as e:
        print(f"  ZMQ error talking to Falcon: {e}")
    finally:
        socket.close()
        terminate(proc)

    # Keep the exact graph that produced this run alongside its data.
    shutil.copy2(out_graph_path, run_results_dir / graph_name)

    if not ok:
        print("  run did not complete cleanly; leaving it in place for inspection.")
        return None

    dest = sweep_dir / run_name
    shutil.move(str(run_results_dir), str(dest))
    print(f"  moved -> {dest}")

    # Falcon repoints ./_last_run at the (now moved) run dir; fix it up.
    last_run = REPO_ROOT / "_last_run"
    if last_run.is_symlink():
        last_run.unlink()
        last_run.symlink_to(dest.resolve())

    time.sleep(1.0)  # let the OS release the ZMQ port before the next run
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="ecHTtests.yaml",
                   help="base graph in resources/graphs/ (default: ecHTtests.yaml)")
    p.add_argument("--snr", type=float, nargs="+", default=[-10, 0, 10, 20, 50],
                   help="SNR levels in dB (default: -10 -5 0 5 10 15 20 50)")
    p.add_argument("--noise", nargs="+", default=["white", "pink"],
                   choices=["white", "pink"], help="noise colours (default: white pink)")
    p.add_argument("--name", default="snr_sweep",
                   help="sweep sub-folder under results/ (default: snr_sweep)")
    p.add_argument("--n-messages", type=int, default=None,
                   help="override SimulatedSource n_messages (run length) for every run")
    p.add_argument("--run-seconds", type=float, default=None,
                   help="stream this many seconds per run before stop/quit "
                        "(default: the graph's own signal length + --settle)")
    p.add_argument("--settle", type=float, default=5.0,
                   help="margin added to the auto run length (default: 5s)")
    p.add_argument("--shutdown", type=float, default=30.0,
                   help="seconds to wait for Falcon to quit after stop (default: 30)")
    p.add_argument("--no-analyse", action="store_true",
                   help="collect the runs but skip the summary plot")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    with open(FALCON_CONFIG) as f:
        falcon_cfg = yaml.safe_load(f)
    port = falcon_cfg["network"]["port"]
    resources_folder = Path(falcon_cfg["server_side_storage"]["resources"])
    if not resources_folder.is_absolute():
        resources_folder = REPO_ROOT / resources_folder

    base_graph_path = resources_folder / "graphs" / args.graph
    out_graph_path  = resources_folder / "graphs" / f"_sweep_{args.graph}"
    sweep_dir       = REPO_ROOT / RESULTS_DIR / args.name

    grid = [(snr, noise) for noise in args.noise for snr in args.snr]
    print(f"Base graph : {base_graph_path}")
    print(f"Sweep dir  : {sweep_dir}")
    print(f"Runs ({len(grid)}): " +
          ", ".join(f"{s:g}dB/{n}" for s, n in grid))
    if args.dry_run:
        return
    if not FALCON_BIN.exists():
        sys.exit(f"Falcon binary not found at {FALCON_BIN} -- build it first.")

    sweep_dir.mkdir(parents=True, exist_ok=True)
    falcon_cfg_backup = FALCON_CONFIG.read_text()
    ctx = zmq.Context()

    completed = []
    try:
        for snr_db, noise_color in grid:
            signal_len = patch_graph(base_graph_path, out_graph_path,
                                     snr_db, noise_color, args.n_messages)
            run_seconds = args.run_seconds if args.run_seconds is not None \
                else signal_len + args.settle
            dest = run_one(ctx, port, args.graph, out_graph_path,
                           snr_db, noise_color, sweep_dir, run_seconds, args.shutdown)
            if dest is not None:
                completed.append(dest)
    except KeyboardInterrupt:
        print("\nInterrupted; keeping the runs completed so far.")
    finally:
        ctx.term()
        out_graph_path.unlink(missing_ok=True)
        FALCON_CONFIG.write_text(falcon_cfg_backup)

    print(f"\n{len(completed)}/{len(grid)} runs completed into {sweep_dir}")
    if completed and not args.no_analyse:
        from analysis.main import analyse_snr_sweep
        analyse_snr_sweep(str(sweep_dir))


if __name__ == "__main__":
    main()
