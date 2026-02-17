"""System-information collection helpers for run artifact snapshots."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any


def _detect_total_memory_bytes() -> int | None:
    if not hasattr(os, "sysconf"):
        return None

    try:
        page_size = None
        for key in ("SC_PAGE_SIZE", "SC_PAGESIZE"):
            if key in os.sysconf_names:
                page_size = os.sysconf(key)
                break

        if page_size is None or "SC_PHYS_PAGES" not in os.sysconf_names:
            return None

        physical_pages = os.sysconf("SC_PHYS_PAGES")
        if (
            isinstance(page_size, int)
            and page_size > 0
            and isinstance(physical_pages, int)
            and physical_pages > 0
        ):
            return page_size * physical_pages
    except (AttributeError, OSError, ValueError):
        return None

    return None


def _detect_nvidia_gpus() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return []

    if result.returncode != 0:
        return []

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        name, memory_total = (line.split(",", maxsplit=1) + [""])[:2]
        gpu: dict[str, Any] = {"name": name.strip()}
        memory_total = memory_total.strip()
        if memory_total:
            try:
                gpu["memory_total_mib"] = int(memory_total)
            except ValueError:
                gpu["memory_total_mib"] = memory_total
        gpus.append(gpu)

    return gpus


def collect_system_info() -> dict[str, Any]:
    """Collect a minimal machine snapshot for experiment provenance."""
    gpus = _detect_nvidia_gpus()
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "count_logical": os.cpu_count(),
            "processor": platform.processor() or None,
        },
        "ram": {
            "total_bytes": _detect_total_memory_bytes(),
        },
        "gpu": {
            "count": len(gpus),
            "nvidia": gpus,
        },
    }
