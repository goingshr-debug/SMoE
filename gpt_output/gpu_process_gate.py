"""Hard gate used by every optimization test and profiler entrypoint."""

import subprocess


def require_process_free_gpus() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = result.stdout.strip()
    if processes:
        raise RuntimeError(
            "Test refused: GPU 0/1 must have zero processes.\n" + processes
        )
