"""Run a closed-loop (CLAS) experiment with Falcon, from start to analysis.

What this script does, in order:
  1. Points Falcon's log file (in .falcon/config.yaml) at a new results folder.
  2. Precomputes the filter coefficients needed by the graph's MultiChannelFilter
     and PhaseEstimation processors (unless the graph gives a coefficient 'file').
  3. Starts the Falcon server and runs the graph. Data is saved to
     results/<results_dir>_<timestamp>/ together with a copy of the graph file.
  4. Lets you control the run from the keyboard (see KEY_HELP) and, optionally,
     plays a timed stimulation on/off protocol read from a CSV file.
  5. When Falcon exits, analyses the last run (see analysis/).

Usage:
    python3 clas.py --graph SimulateCLAS.yaml --results_dir my_experiment
"""

import argparse
import csv
import os
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import traceback
import tty
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import zmq

from analysis.main import analyse_results
from gen_filter_coeff import gen_filter, gen_filter_ecHT


REPO_ROOT = Path(__file__).resolve().parent
FALCON_BINARY = "./build/release/falcon/falcon"
FALCON_CONFIG = REPO_ROOT / ".falcon" / "config.yaml"
FALCON_STARTUP_WAIT_S = 0.5

RESULTS_DIR = "results"       # Falcon writes each run into RESULTS_DIR/<run name>
LAST_RUN_DIR = "_last_run/"   # symlink Falcon points at the most recent run

STIM_PROCESSOR_NAME = "StimControl"
STIM_AUTOSTART_DELAY_S = 15.0

# PhaseEstimation needs one bandpass filter per candidate f0: centred on f0,
# 90% of f0 wide, with a window length of 2 cycles of f0.
PHASE_FILTER_F0_GRID = np.arange(4.9, 18.1, 0.1)  # Hz
PHASE_FILTER_ORDER = 1
PHASE_FILTER_REL_BANDWIDTH = 0.9
PHASE_FILTER_CYCLES = 2.0

KEY_HELP = """
Keyboard controls (no Enter needed):
  s  stop the graph (and the stimulation protocol, if running)
  r  restart the graph, recording into a new timestamped results folder
  z  start the stimulation protocol now
  q  quit Falcon; the analysis runs afterwards
  h  show this help
"""


# --- Setup -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a CLAS experiment with Falcon and analyse the result.")
    parser.add_argument("--graph", default="TurboLinkCLAS.yaml",
                        help="graph file name, looked up in <resources>/graphs/ (default: %(default)s)")
    parser.add_argument("--results_dir",
                        help="name for this run; a timestamp is appended. Asked for if omitted.")
    parser.add_argument("--stim_protocol", default="resources/stim_protocol.csv",
                        help="CSV with 'duration_s,state' rows, relative to the repo root (default: %(default)s)")
    return parser.parse_args()


def make_run_name(base_name):
    """'my_experiment' -> 'my_experiment_20261021_143000'."""
    return f"{Path(base_name).name}_{time.strftime('%Y%m%d_%H%M%S')}"


def update_falcon_config(log_path):
    """Point Falcon's log file at log_path and return the full Falcon config."""
    with open(FALCON_CONFIG, "r+") as f:
        config = yaml.safe_load(f)
        config["logging"]["path"] = str(log_path)
        f.seek(0)
        yaml.dump(config, f)
        f.truncate()
    return config


def precompute_filters(graph_config, filters_folder):
    """Generate coefficient files for filters the graph defines by parameters
    (rather than by a 'file' entry), so Falcon finds them when it starts."""
    graph = graph_config.get("graph", {})
    fs = graph.get("defaults", {}).get("fs")

    for processor_config in graph.get("processors", {}).values():
        filter_config = processor_config.get("options", {}).get("filter", {})
        if "file" in filter_config:
            continue  # coefficients come from an existing file
        if processor_config.get("class") == "MultiChannelFilter":
            precompute_multichannel_filter(filter_config, fs, filters_folder)
        elif processor_config.get("class") == "PhaseEstimation":
            precompute_phase_estimation_filters(fs, filters_folder)


def precompute_multichannel_filter(filter_config, fs, filters_folder):
    """A cascade of Butterworth stages, given as parallel lists in the graph:
    N, low_cutoff, high_cutoff and (optionally) btype, one entry per stage."""
    N = filter_config.get("N")
    btype = filter_config.get("btype", "bandpass")
    if isinstance(btype, str):  # a single btype applies to every stage
        btype = [btype] * len(N)
    filter_params = list(zip(N, filter_config.get("low_cutoff"), filter_config.get("high_cutoff"), btype))
    gen_filter(filter_params, fs, filter_config.get("name"), output_folder=filters_folder)


def precompute_phase_estimation_filters(fs, filters_folder):
    """One bandpass frequency response per candidate f0 (existing files are skipped)."""
    for f0 in PHASE_FILTER_F0_GRID:
        bandwidth = PHASE_FILTER_REL_BANDWIDTH * f0
        filter_length = int(PHASE_FILTER_CYCLES * fs / f0)
        params = [(PHASE_FILTER_ORDER, f0 - bandwidth / 2, f0 + bandwidth / 2, fs, filter_length, "bandpass", f0)]
        gen_filter_ecHT(params, output_folder=filters_folder)


# --- Talking to Falcon -------------------------------------------------------

def connect_to_falcon(zmq_context, port):
    socket = zmq_context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{port}")
    return socket


def falcon_request(socket, *frames):
    """Send one command to Falcon and wait for its reply (a REQ socket must
    alternate send and receive)."""
    socket.send_multipart([frame if isinstance(frame, bytes) else frame.encode() for frame in frames])
    return socket.recv_multipart()


def start_recording(socket, run_name, graph_path):
    """Start the graph, saving into RESULTS_DIR/run_name, and keep a copy of the
    graph file there for record-keeping. Returns the results folder."""
    falcon_request(socket, "graph", "start", RESULTS_DIR, run_name, "")
    results_path = Path(RESULTS_DIR) / run_name
    results_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(graph_path, results_path / Path(graph_path).name)
    return results_path


def terminate(proc):
    """Stop Falcon's process group, escalating SIGINT -> SIGTERM -> SIGKILL.
    SIGINT comes first because Falcon handles it like Ctrl+C (a clean shutdown);
    SIGTERM is not handled and kills it immediately. Falcon is started with
    start_new_session=True, so this never touches this script itself."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue  # still running: try the next, stronger signal
        except Exception:
            return


# --- Stimulation protocol ----------------------------------------------------

def load_stim_protocol(path):
    """Load a stimulation protocol from a CSV with columns 'duration_s' and 'state'.

    'state' is on/off (case-insensitive); 'duration_s' accepts a trailing 's' (e.g. '10s').
    Returns a list of (duration_seconds, enabled) tuples.
    """
    protocol = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            duration_s = float(row["duration_s"].strip().rstrip("sS"))
            state = row["state"].strip().lower()
            if state not in ("on", "off"):
                raise ValueError(f"Invalid state '{row['state']}', expected 'on' or 'off'.")
            protocol.append((duration_s, state == "on"))
    return protocol


def run_stim_protocol(zmq_context, port, protocol, log_path, stop_event: threading.Event):
    """Runs a stimulation protocol on its own REQ socket (zmq sockets are not thread-safe,
    so this must not share the main thread's socket), toggling StimControl's gain
    on/off via the "set_enabled" apply command and logging each transition with a timestamp.
    """
    socket = connect_to_falcon(zmq_context, port)

    try:
        start_time = datetime.now()
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "elapsed_s", "state"])
            writer.writerow([start_time.isoformat(), 0.0, "protocol_start"])
            f.flush()

            for duration_s, enabled in protocol:
                if stop_event.is_set():
                    break

                state_str = "on" if enabled else "off"
                # Sends e.g. "{StimControl: {set_enabled: {enabled: true}}}"
                apply_cmd = yaml.safe_dump({STIM_PROCESSOR_NAME: {"set_enabled": {"enabled": enabled}}},
                                           default_flow_style=True).strip()
                falcon_request(socket, "graph", "apply", apply_cmd)

                elapsed = (datetime.now() - start_time).total_seconds()
                writer.writerow([datetime.now().isoformat(), f"{elapsed:.3f}", state_str])
                f.flush()
                print(f"[stim protocol] {state_str} for {duration_s}s")

                if stop_event.wait(duration_s):
                    break

            end_state = "protocol_stopped" if stop_event.is_set() else "protocol_end"
            elapsed = (datetime.now() - start_time).total_seconds()
            writer.writerow([datetime.now().isoformat(), f"{elapsed:.3f}", end_state])
            print(f"[stim protocol] {end_state}")

            if not stop_event.is_set():
                # Protocol ran to completion (not manually stopped via 's'/'q'):
                # stop the graph and terminate Falcon so main() can proceed
                # straight to analyse_results without waiting for a keypress.
                print("[stim protocol] Protocol complete, stopping graph and terminating Falcon...")
                falcon_request(socket, "graph", "stop")
                falcon_request(socket, "quit")
    finally:
        socket.close()


class StimProtocolRunner:
    """Plays the stimulation protocol in a background thread, so the keyboard
    stays responsive while it runs."""

    def __init__(self, zmq_context, port, protocol_path):
        self.zmq_context = zmq_context
        self.port = port
        self.protocol_path = protocol_path
        self.results_path = None  # where the protocol and its log are saved; updated on restart
        self._thread = None
        self._stop_event = threading.Event()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running():
            print("Stimulation protocol already running; press 's' or 'q' to stop it early.")
            return
        try:
            protocol = load_stim_protocol(self.protocol_path)
        except (OSError, ValueError, KeyError) as e:
            print(f"Failed to load stim protocol from {self.protocol_path}: {e}")
            return
        shutil.copy2(self.protocol_path, self.results_path / self.protocol_path.name)
        log_path = self.results_path / f"stim_protocol_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=run_stim_protocol,
            args=(self.zmq_context, self.port, protocol, log_path, self._stop_event),
            daemon=True,
        )
        self._thread.start()
        print(f"Started stimulation protocol from {self.protocol_path} ({len(protocol)} steps).")

    def start_after(self, delay_s, falcon_process):
        """Start automatically after delay_s, unless stop() is called first or Falcon has exited."""
        def wait_then_start():
            if self._stop_event.wait(delay_s):
                return
            if falcon_process.poll() is not None:
                return
            self.start()

        threading.Thread(target=wait_then_start, daemon=True).start()

    def stop(self):
        """Stop the protocol if it is running, and cancel any pending automatic start."""
        self._stop_event.set()
        if self.is_running():
            self._thread.join()


# --- Keyboard ----------------------------------------------------------------

def get_char(timeout=None):
    """Read a single keypress. If timeout is given, returns None if no key
    is pressed within that many seconds instead of blocking indefinitely."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if timeout is not None:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return None
        ch = sys.stdin.read(1)  # returns immediately on any keypress
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # restore terminal
    return ch


# --- Main --------------------------------------------------------------------

def main():
    args = parse_args()
    if not args.results_dir:
        args.results_dir = input("Enter the results directory: ")

    # 1. Point Falcon's log at the new results folder and read the graph
    run_name = make_run_name(args.results_dir)
    config = update_falcon_config(log_path=Path(RESULTS_DIR) / run_name / f"{run_name}.log")
    port = config["network"]["port"]
    resources_folder = config.get("server_side_storage", {}).get("resources", "")

    graph_path = os.path.join(resources_folder, "graphs", args.graph)
    print(f"Using graph file: {graph_path}")
    with open(graph_path, "r") as f:
        graph_config = yaml.safe_load(f)
    use_stim_protocol = graph_config.get("graph", {}).get("defaults", {}).get("use_stim_protocol", False)

    # 2. Make sure every filter the graph needs has a coefficient file
    precompute_filters(graph_config, filters_folder=os.path.join(resources_folder, "filters"))

    # 3. Start Falcon and the recording
    falcon_process = subprocess.Popen(
        [FALCON_BINARY, args.graph, "--config", FALCON_CONFIG],
        stdin=subprocess.PIPE,
        start_new_session=True,  # own process group, so terminate() only kills Falcon
    )
    time.sleep(FALCON_STARTUP_WAIT_S)
    context = zmq.Context()
    socket = connect_to_falcon(context, port)
    stim = StimProtocolRunner(context, port, REPO_ROOT / args.stim_protocol)

    try:
        stim.results_path = start_recording(socket, run_name, graph_path)

        if use_stim_protocol:
            stim.start_after(STIM_AUTOSTART_DELAY_S, falcon_process)
            print(f"\nuse_stim_protocol is enabled; stimulation protocol will start automatically "
                  f"in {STIM_AUTOSTART_DELAY_S:g}s (press 'z' to start it sooner).")
        print(KEY_HELP)

        # 4. React to keypresses until Falcon exits (after 'q', or when the stim protocol completes)
        while falcon_process.poll() is None:
            key = get_char(timeout=0.5)
            if key is None:
                continue
            key = key.strip().lower()

            if key == "s":
                stim.stop()
                falcon_request(socket, "graph", "stop")
                time.sleep(1)
                print("\nGraph stopped. Press 'q' to quit Falcon and run the analysis, "
                      "or 'r' to start a new recording.")
            elif key == "r":
                stim.results_path = start_recording(socket, make_run_name(args.results_dir), graph_path)
            elif key == "z":
                stim.start()
            elif key == "q":
                stim.stop()
                falcon_request(socket, "quit")
            elif key == "h":
                print(KEY_HELP)

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during run: {e}")
        traceback.print_exc()
    finally:
        stim.stop()
        falcon_process.stdin.close()
        terminate(falcon_process)
        socket.close()
        context.term()

    # 5. Analyse the run that just finished
    analyse_results(LAST_RUN_DIR)


if __name__ == "__main__":
    main()
