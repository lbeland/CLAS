import argparse
import os
import signal
import numpy as np
import time
import subprocess
from pathlib import Path
import yaml
from gen_filter_coeff import gen_bandpass
from plot_results import plot_results


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_FALCON_CONFIG = REPO_ROOT / ".falcon" / "config.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="SimulateCLAS.yaml")
    args = parser.parse_args()
    GRAPH_CONFIG = args.graph

    # Load the Falcon configuration
    with open(WORKSPACE_FALCON_CONFIG, "r") as f:
        config = yaml.safe_load(f)
        server_config = config.get("server_side_storage", {})
        resources_folder = server_config.get("resources", "")

    graph_path = os.path.join(resources_folder, "graphs", GRAPH_CONFIG)
    print(f"Using graph file: {graph_path}")

    with open(graph_path, "r") as f:
        graph_config = yaml.safe_load(f)
        fs = graph_config.get("graph", {}).get("defaults", {}).get("fs", None)  # Default to 1000 Hz if not specified
        
        for processor in graph_config.get("graph", {}).get("processors", []):
            processor_config = graph_config.get("graph", {}).get("processors", {}).get(processor, {})
            if processor_config.get("class") == "MultiChannelFilter":
                # Check if processor MultiChannelFilter is configured to use a non-file-based filter
                filter_config = processor_config.get("options", {}).get("filter", {})
                if "file" not in filter_config:
                    N = filter_config.get("N", 1)
                    low_cutoff = filter_config.get("low_cutoff")
                    high_cutoff = filter_config.get("high_cutoff")
                    gen_bandpass(N, low_cutoff, high_cutoff, fs, length=None, output_folder=os.path.join(resources_folder, "filters"))
            elif processor_config.get("class") == "PhaseEstimator":
                # Check if processor PhaseEstimator is configured to use a non-file-based filter
                filter_config = processor_config.get("options", {}).get("filter", {})
                if "file" not in filter_config:
                    for iaf in np.arange(4.9,18.1,0.1):
                        bandwidth = filter_config.get("bandwidth", 4)
                        filter_length = int(2.0 * fs/iaf)    # 2 cycles of iaf frequency
                        gen_bandpass(1, iaf-bandwidth/2, iaf+bandwidth/2, fs, filter_length, output_folder=os.path.join(resources_folder, "filters"))


    graph_process = subprocess.Popen(["sudo", "-E", "chrt", "-f", "99","./build/release/falcon/falcon", GRAPH_CONFIG, "--config", WORKSPACE_FALCON_CONFIG, "--autostart"]) 
    try:
        # Wait for falcon to complete (processors will auto-exit after processing n_messages)
        return_code = graph_process.wait()
        print(f"Falcon process exited with code {return_code}")
    except subprocess.TimeoutExpired:
        print("Timeout: Falcon did not complete within Timeout. Terminating...")
        terminate(graph_process)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        terminate(graph_process)

    # Plot results
    plot_results(fs, 7.5, GRAPH_CONFIG)


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
