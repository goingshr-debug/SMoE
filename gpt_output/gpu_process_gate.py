"""Hard gate used by every optimization test and profiler entrypoint."""

import csv
import io
import subprocess


_AUTHORIZED_MONITOR_PID = "1393879"
_AUTHORIZED_MONITOR_NAME = "/isaac-sim/kit/python/bin/python3"


def _unauthorized_processes(output: str) -> list[list[str]]:
    processes = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not row:
            continue
        normalized = [field.strip() for field in row]
        if len(normalized) < 4:
            processes.append(normalized)
            continue
        _, pid, process_name, _ = normalized[:4]
        if pid == _AUTHORIZED_MONITOR_PID and process_name == _AUTHORIZED_MONITOR_NAME:
            continue
        processes.append(normalized)
    return processes


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
    unauthorized = _unauthorized_processes(result.stdout)
    if unauthorized:
        processes = "\n".join(", ".join(row) for row in unauthorized)
        raise RuntimeError(
            "Test refused: GPU 0/1 contain an unauthorized compute process.\n"
            + processes
        )
