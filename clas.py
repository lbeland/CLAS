"""Spawn mockup streamer (producer) and threshold controller (consumer) directly via their mains, benchmark IPC."""
import argparse
import os
import signal
import time
import subprocess
from pathlib import Path
import yaml


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_FALCON_CONFIG = REPO_ROOT / ".falcon" / "config.yaml"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_messages", type=int, default=2000)
    parser.add_argument("--msg_size", type=int, default=1)
    parser.add_argument("--num_channels", type=int, default=10)
    parser.add_argument("--max_buffer_size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output_file", type=str, default="Producer.csv")
    args = parser.parse_args()

    results_folder = "rt_c_results"

    os.makedirs(results_folder, exist_ok=True)

    # 1) Start consumer first
    output_path = str(REPO_ROOT / results_folder / f"{args.num_channels}_{args.msg_size}")
   
    if os.path.exists(output_path + "_Consumer.csv") and not args.overwrite:
        return

    graph_process = subprocess.Popen(["sudo", "-E", "chrt", "-f", "99","./build/falcon/falcon", "SimulateCLAS.yaml", "--config", WORKSPACE_FALCON_CONFIG, "--autostart"]) 
    try:
        # 3) Wait for falcon to complete (processors will auto-exit after processing n_messages)
        return_code = graph_process.wait()
        print(f"Falcon process exited with code {return_code}")
    except subprocess.TimeoutExpired:
        print("Timeout: Falcon did not complete within 10 seconds. Terminating...")
        terminate(graph_process)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        terminate(graph_process)


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
