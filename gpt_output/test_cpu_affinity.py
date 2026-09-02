#!/usr/bin/env python3
"""Host-topology checks for the production CPU placement policy."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cpu_affinity import select_cpu_placement


def topology(cpu: int) -> tuple[int, int]:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return (
        int((root / "physical_package_id").read_text()),
        int((root / "core_id").read_text()),
    )


for total_cores in (2, 3, 5, 9, 17):
    placement = select_cpu_placement(total_cores, sample_interval=0.01)
    assert len(placement.compute_cores) == total_cores - 1
    assert placement.shared_core not in placement.compute_cores
    physical = [topology(cpu) for cpu in placement.compute_cores]
    assert len(physical) == len(set(physical)), (total_cores, placement, physical)
    if total_cores <= 17:
        assert len(placement.compute_packages) == 1, placement
    if total_cores <= 16:
        assert topology(placement.shared_core) not in set(physical), placement
    print(total_cores, placement)
