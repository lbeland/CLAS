import argparse
import csv
import os
import signal
import threading
import numpy as np
import time
import subprocess
from datetime import datetime
from pathlib import Path
import shutil
import yaml
import zmq
from gen_filter_coeff import gen_filter, gen_filter_ecHT
from analysis.main import analyse_results


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_FALCON_CONFIG = REPO_ROOT / ".falcon" / "config.yaml"
RESULTS_DIR = "results"
STIM_PROCESSOR_NAME = "StimControl"

import tty
import termios
import sys
import select

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="TurboLinkCLAS.yaml")
    parser.add_argument("--results_dir")
    parser.add_argument("--stim_protocol", default="stim_protocol.csv")
    args = parser.parse_args()

    stim_protocol_path = REPO_ROOT / args.stim_protocol

    if not args.results_dir:
        args.results_dir = input("Enter the results directory: ")

    output = f"{Path(args.results_dir).name}_{time.strftime('%Y%m%d_%H%M%S')}"
    # Copy graph file to results directory for record-keeping
    results_dir_path =  Path(RESULTS_DIR) / Path(output)

    # Load the Falcon configuration
    with open(WORKSPACE_FALCON_CONFIG, "r+") as f:
        config = yaml.safe_load(f)
        server_config = config.get("server_side_storage", {})
        resources_folder = server_config.get("resources", "")
        # Set the logging path in the configuration to the results directory
        config["logging"]["path"] = str(results_dir_path / f"{output}.log")
        # Save the updated configuration back to the file
        f.seek(0)
        yaml.dump(config, f)
        f.truncate()

    graph_path = os.path.join(resources_folder, "graphs", args.graph)
    print(f"Using graph file: {graph_path}")

    with open(graph_path, "r") as f:
        graph_config = yaml.safe_load(f)
        fs = graph_config.get("graph", {}).get("defaults", {}).get("fs", None)
        use_stim_protocol = graph_config.get("graph", {}).get("defaults", {}).get("use_stim_protocol", False)
        
        # Iterate through processors in the graph to find filter configurations and precompute filter coefficients if needed
        for processor in graph_config.get("graph", {}).get("processors", []):
            processor_config = graph_config.get("graph", {}).get("processors", {}).get(processor, {})
            if processor_config.get("class") == "MultiChannelFilter":
                # Check if processor MultiChannelFilter is configured to use a non-file-based filter
                filter_config = processor_config.get("options", {}).get("filter", {})
                if "file" not in filter_config:
                    N = filter_config.get("N")
                    low_cutoff = filter_config.get("low_cutoff")
                    high_cutoff = filter_config.get("high_cutoff")
                    btype = filter_config.get("btype", "bandpass")
                    filter_params = list(zip(N, low_cutoff, high_cutoff, btype))
                    filter_name = filter_config.get("name")
                    gen_filter(filter_params, fs, filter_name, output_folder=os.path.join(resources_folder, "filters"))
                    # gen_filter(N, low_cutoff, high_cutoff, fs, length=None, output_folder=os.path.join(resources_folder, "filters"), btype=btype)
            elif processor_config.get("class") == "PhaseEstimation":
                # Check if processor PhaseEstimation is configured to use a non-file-based filter
                filter_config = processor_config.get("options", {}).get("filter", {})
                if "file" not in filter_config:
                    for f0 in np.arange(4.9,18.1,0.1):
                        # bandwidth = filter_config.get("bandwidth", 4)
                        bandwidth = 0.9 * f0
                        filter_length = int(2.0 * fs/f0)    # 2 cycles of f0 frequency
                        params = [(1, f0-bandwidth/2, f0+bandwidth/2, fs, filter_length, "bandpass", f0)]
                        gen_filter_ecHT(params, output_folder=os.path.join(resources_folder, "filters"))
     

    try:

        graph_process = subprocess.Popen(["./build/release/falcon/falcon",args.graph, "--config", WORKSPACE_FALCON_CONFIG],stdin=subprocess.PIPE)

        time.sleep(0.5)

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://127.0.0.1:{config['network']['port']}")
        socket.send_multipart([b"graph", b"start", RESULTS_DIR.encode(), output.encode(), b""])
        socket.recv_multipart()

        results_dir_path.mkdir(parents=True, exist_ok=True)
        # Copy the graph file to the results directory for record-keeping
        output_graph_path = results_dir_path / args.graph
        shutil.copy2(graph_path, output_graph_path)

        # Wait for falcon to complete (processors will auto-exit after processing n_messages)
        # return_code = graph_process.wait()

        stim_thread = None
        stim_stop_event = threading.Event()

        def start_stim_protocol():
            nonlocal stim_thread
            if stim_thread is not None and stim_thread.is_alive():
                print("Stimulation protocol already running; press 's' or 'q' to stop it early.")
                return
            try:
                protocol = load_stim_protocol(stim_protocol_path)
            except (OSError, ValueError, KeyError) as e:
                print(f"Failed to load stim protocol from {stim_protocol_path}: {e}")
                return
            shutil.copy2(stim_protocol_path, results_dir_path / stim_protocol_path.name)
            stim_log_path = results_dir_path / f"stim_protocol_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            stim_stop_event.clear()
            stim_thread = threading.Thread(
                target=run_stim_protocol,
                args=(context, config["network"]["port"], protocol, stim_log_path, stim_stop_event),
                daemon=True,
            )
            stim_thread.start()
            print(f"Started stimulation protocol from {stim_protocol_path} ({len(protocol)} steps).")

        if use_stim_protocol:
            def auto_start_stim_protocol():
                if stim_stop_event.wait(15.0):
                    return
                if graph_process.poll() is not None:
                    return
                start_stim_protocol()

            threading.Thread(target=auto_start_stim_protocol, daemon=True).start()
            print("\n use_stim_protocol is enabled; stimulation protocol will start automatically in 15s (press 'z' to start it sooner).")

        while graph_process.poll() is None:

            command = get_char(timeout=0.5)
            if command is None:
                continue
            if command.strip().lower() == "s":
                if stim_thread is not None and stim_thread.is_alive():
                    stim_stop_event.set()
                    stim_thread.join()
                socket.send_multipart([b"graph", b"stop"])
                socket.recv_multipart()
            elif command.strip().lower() == "r":
                output = f"{Path(args.results_dir).name}_{time.strftime('%Y%m%d_%H%M%S')}"
                socket.send_multipart([b"graph", b"start", RESULTS_DIR.encode(), output.encode(), b""])
                socket.recv_multipart()
                # Copy graph file to results directory for record-keeping
                results_dir_path =  Path(RESULTS_DIR) / Path(output)
                results_dir_path.mkdir(parents=True, exist_ok=True)
                output_graph_path = results_dir_path / args.graph
                shutil.copy2(graph_path, output_graph_path)
            elif command.strip().lower() == "q":
                if stim_thread is not None and stim_thread.is_alive():
                    stim_stop_event.set()
                    stim_thread.join()
                socket.send_multipart([b"quit"])
                socket.recv_multipart()
            elif command.strip().lower() == "z":
                start_stim_protocol()

        if stim_thread is not None and stim_thread.is_alive():
            stim_stop_event.set()
            stim_thread.join()

        # print(f"Falcon process exited with code {return_code}")
    except subprocess.TimeoutExpired:
        print("Timeout: Falcon did not complete within Timeout. Terminating...")
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
    
    graph_process.stdin.close()
    terminate(graph_process)
    
    socket.close()
    context.term()

    # Postprocessing results
    analyse_results(10, "_last_run/")
    # plot_results(fs, 7.5, GRAPH_CONFIG)

def terminate(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass

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
    socket = zmq_context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{port}")

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
                socket.send_multipart([
                    b"graph", b"apply",
                    (f"{{{STIM_PROCESSOR_NAME}: {{set_enabled: "
                     f"{{enabled: {str(enabled).lower()}}}}}}}").encode(),
                ])
                socket.recv_multipart()

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
                socket.send_multipart([b"graph", b"stop"])
                socket.recv_multipart()
                socket.send_multipart([b"quit"])
                socket.recv_multipart()
    finally:
        socket.close()


if __name__ == "__main__":
    main()
