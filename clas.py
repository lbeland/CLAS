import argparse
import os
import signal
import numpy as np
import time
import subprocess
from pathlib import Path
import shutil
import yaml
import zmq
from gen_filter_coeff import gen_filter, gen_filter_ecHT
from plot_results import plot_results
from postprocess_results import analyse_results


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_FALCON_CONFIG = REPO_ROOT / ".falcon" / "config.yaml"
RESULTS_DIR = "results"

import tty
import termios
import sys

def get_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)  # returns immediately on any keypress
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # restore terminal
    return ch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="TurboLinkCLAS.yaml")
    parser.add_argument("--results_dir")
    args = parser.parse_args()

    if not args.results_dir:
        args.results_dir = input("Enter the results directory: ")

    # Load the Falcon configuration
    with open(WORKSPACE_FALCON_CONFIG, "r") as f:
        config = yaml.safe_load(f)
        server_config = config.get("server_side_storage", {})
        resources_folder = server_config.get("resources", "")

    graph_path = os.path.join(resources_folder, "graphs", args.graph)
    print(f"Using graph file: {graph_path}")

    filter_params = []

    with open(graph_path, "r") as f:
        graph_config = yaml.safe_load(f)
        fs = graph_config.get("graph", {}).get("defaults", {}).get("fs", None)
        
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
                    # gen_filter(N, low_cutoff, high_cutoff, fs, length=None, output_folder=os.path.join(resources_folder, "filters"), btype=btype)
            elif processor_config.get("class") == "PhaseEstimator":
                # Check if processor PhaseEstimator is configured to use a non-file-based filter
                filter_config = processor_config.get("options", {}).get("filter", {})
                if "file" not in filter_config:
                    for iaf in np.arange(4.9,18.1,0.1):
                        bandwidth = filter_config.get("bandwidth", 4)
                        filter_length = int(2.0 * fs/iaf)    # 2 cycles of iaf frequency
                        params = [(1, iaf-bandwidth/2, iaf+bandwidth/2, fs, filter_length, "bandpass")]
                        gen_filter_ecHT(params, output_folder=os.path.join(resources_folder, "filters"))

    if filter_params:
        gen_filter(filter_params, fs, output_folder=os.path.join(resources_folder, "filters"))



    graph_process = subprocess.Popen(["./build/release/falcon/falcon",args.graph, "--config", WORKSPACE_FALCON_CONFIG],stdin=subprocess.PIPE)

    time.sleep(0.5)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{config['network']['port']}")
    output = f"{Path(args.results_dir).name}_{time.strftime('%Y%m%d_%H%M%S')}"
    socket.send_multipart([b"graph", b"start", RESULTS_DIR.encode(), output.encode(), b""])
    socket.recv_multipart()
    # Copy graph file to results directory for record-keeping
    results_dir_path =  Path(RESULTS_DIR) / Path(output)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    output_graph_path = results_dir_path / args.graph
    shutil.copy2(graph_path, output_graph_path)

    try:
        # Wait for falcon to complete (processors will auto-exit after processing n_messages)
        # return_code = graph_process.wait()

        while graph_process.poll() is None:
            
            command = get_char()
            if command.strip().lower() == "s":
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
                socket.send_multipart([b"quit"])
                socket.recv_multipart()
            time.sleep(0.5)

        # print(f"Falcon process exited with code {return_code}")
    except subprocess.TimeoutExpired:
        print("Timeout: Falcon did not complete within Timeout. Terminating...")
        terminate(graph_process)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        terminate(graph_process)

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


if __name__ == "__main__":
    main()
