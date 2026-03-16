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
# PRODUCER_ROOT = REPO_ROOT / "c-producer"
# CONSUMER_ROOT = REPO_ROOT / "c-consumer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_messages", type=int, default=1000000)
    parser.add_argument("--msg_size", type=int, default=1)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--max_buffer_size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output_file", type=str, default="Producer.csv")
    args = parser.parse_args()

    results_folder = "rt_c_results"

    os.makedirs(results_folder, exist_ok=True)

    # 1) Start consumer first
    output_path = str(REPO_ROOT / results_folder / f"{args.num_channels}_{args.msg_size}")
   
    if os.path.exists(output_path + "_Consumer.csv") and not args.overwrite:
        return
    

    graph = {
        "graph": {
        "name": "CLAS",
        "processors": {
            "Producer": {
                "class": "Producer",
                "options": {
                    "nchannels": args.num_channels,
                    "nsamples": args.msg_size,
                    "n_messages": args.n_messages,
                    "output_file": str(output_path)
                },
                "advanced":{
                    "buffer_sizes": {
                        "out": args.max_buffer_size
                    },
                    "thread_core": 7
                }
            },
            "PhaseEstimator": {
                "class": "PhaseEstimator",
                "options": {
                    "n_messages": args.n_messages
                },
                "advanced":{
                    "buffer_sizes": {
                        "out": args.max_buffer_size
                    },
                    "thread_core": 8
                }
            },
            "Consumer": {
                "class": "Consumer",
                "options": {
                    "n_messages": args.n_messages,
                    "output_file": str(output_path)
                },
                "advanced":{
                    "thread_core": 9
                }
            }
        },
        "connections": [
            "Producer.out.0 = PhaseEstimator.in.0",
            "PhaseEstimator.out.0 = Consumer.in.0"
        ]
    }
    }

    with open("resources/graphs/simple_test.yaml", "w") as f:
        yaml.safe_dump(graph, f, sort_keys=False)

    graph_process = subprocess.Popen(["sudo", "-E", "chrt", "-f", "99","./build/falcon/falcon", "simple_test.yaml", "--config", WORKSPACE_FALCON_CONFIG, "--autostart"]) 

    try:
        # 3) Wait for falcon to complete (processors will auto-exit after processing n_messages)
        return_code = graph_process.wait()#timeout=20)
        print(f"Falcon process exited with code {return_code}")
    except subprocess.TimeoutExpired:
        print("Timeout: Falcon did not complete within 10 seconds. Terminating...")
        terminate(graph_process)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        terminate(graph_process)
    finally:
        print("Benchmark complete")


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
