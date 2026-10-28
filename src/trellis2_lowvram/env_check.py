"""Environment checks for install and `doctor`.

Hardware checks use NVML (nvidia-ml-py) and psutil only, so they run before torch is installed.
Software checks (torch, CUDA extensions, attention backends) run only when torch is importable.
Every result carries i18n keys for "what happened" and "what to do"; no user text lives here.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GIB = 1024**3

# Thresholds. VRAM/RAM cards report a little under their marketing size (16GB -> ~15.99 GiB).
VRAM_OK_GIB = 15.0
RAM_OK_GIB = 30.0
RAM_MIN_GIB = 15.0
# Estimate, not measured: models for the 512 path ~12GB, + 1024 models ~5GB, + headroom.
DISK_OK_GIB = 30.0
DISK_MIN_GIB = 20.0
MIN_COMPUTE_CAPABILITY = (8, 0)  # Ampere = RTX 30. Turing (RTX 20, 7.5) is unsupported.
MIN_DRIVER_CUDA = 12080  # cu128 wheels need a driver that supports CUDA 12.8 (R570+)
WINDOWS_PATH_MAX = 80  # deep paths inside site-packages hit the 260-char limit; leave room

DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class Check:
    id: str
    status: Status
    msg: str  # i18n key: what happened (and why, when known)
    fix: str | None = None  # i18n key: what the user should do
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class GpuInfo:
    name: str
    vram_gib: float
    compute_capability: tuple[int, int]
    driver_version: str
    driver_cuda: int  # e.g. 12080 for 12.8


# ---------------------------------------------------------------- probes (side effects only here)


def probe_gpus() -> list[GpuInfo] | None:
    """None means NVML itself is unavailable (no NVIDIA driver)."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception as e:  # NVMLError_LibraryNotFound, DriverNotLoaded, ...
        log.debug("nvmlInit failed: %r", e)
        return None
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver_cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            gpus.append(
                GpuInfo(
                    name=name.decode() if isinstance(name, bytes) else name,
                    vram_gib=mem.total / GIB,
                    compute_capability=(int(cc[0]), int(cc[1])),
                    driver_version=driver.decode() if isinstance(driver, bytes) else driver,
                    driver_cuda=int(driver_cuda),
                )
            )
        return gpus
    finally:
        pynvml.nvmlShutdown()


def probe_ram_gib() -> float:
    import psutil

    return psutil.virtual_memory().total / GIB


def probe_disk_free_gib(path: Path) -> float:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / GIB


# ---------------------------------------------------------------- judgements (pure, unit-tested)


def check_os(system: str) -> Check:
    if system in ("Windows", "Linux"):
        return Check("os", Status.OK, "check.os.ok", params={"os": system})
    return Check("os", Status.FAIL, "check.os.unsupported", "fix.os.unsupported", {"os": system})


def check_gpu(gpus: list[GpuInfo] | None) -> list[Check]:
    if not gpus:
        return [Check("gpu", Status.FAIL, "check.gpu.missing", "fix.gpu.missing")]
    # Use the GPU with the most memory; CUDA_VISIBLE_DEVICES can be set by advanced users.
    g = max(gpus, key=lambda x: x.vram_gib)
    p = {
        "name": g.name,
        "vram": f"{g.vram_gib:.1f}",
        "cc": f"{g.compute_capability[0]}.{g.compute_capability[1]}",
        "driver": g.driver_version,
        "driver_cuda": f"{g.driver_cuda // 1000}.{(g.driver_cuda % 1000) // 10}",
        "url": DRIVER_URL,
    }
    out = []
    if g.compute_capability < MIN_COMPUTE_CAPABILITY:
        out.append(Check("gpu", Status.FAIL, "check.gpu.too_old", "fix.gpu.too_old", p))
    else:
        out.append(Check("gpu", Status.OK, "check.gpu.ok", params=p))

    if g.vram_gib >= VRAM_OK_GIB:
        out.append(Check("vram", Status.OK, "check.vram.ok", params=p))
    else:
        out.append(Check("vram", Status.WARN, "check.vram.low", "fix.vram.low", p))

    if g.driver_cuda >= MIN_DRIVER_CUDA:
        out.append(Check("driver", Status.OK, "check.driver.ok", params=p))
    else:
        out.append(Check("driver", Status.FAIL, "check.driver.old", "fix.driver.old", p))
    return out


def check_ram(total_gib: float) -> Check:
    p = {"ram": f"{total_gib:.0f}"}
    if total_gib >= RAM_OK_GIB:
        return Check("ram", Status.OK, "check.ram.ok", params=p)
    if total_gib >= RAM_MIN_GIB:
        return Check("ram", Status.WARN, "check.ram.low", "fix.ram.low", p)
    return Check("ram", Status.FAIL, "check.ram.too_low", "fix.ram.too_low", p)


def check_disk(free_gib: float, where: Path) -> Check:
    p = {"free": f"{free_gib:.0f}", "need": f"{DISK_OK_GIB:.0f}", "path": str(where)}
    if free_gib >= DISK_OK_GIB:
        return Check("disk", Status.OK, "check.disk.ok", params=p)
    if free_gib >= DISK_MIN_GIB:
        return Check("disk", Status.WARN, "check.disk.low", "fix.disk.low", p)
    return Check("disk", Status.FAIL, "check.disk.too_low", "fix.disk.low", p)


def check_install_path(path: Path, system: str) -> list[Check]:
    s = str(path)
    p = {"path": s, "max": WINDOWS_PATH_MAX}
    out = []
    if not s.isascii():
        out.append(Check("path_ascii", Status.WARN, "check.path.non_ascii", "fix.path.move", p))
    if " " in s:
        out.append(Check("path_space", Status.WARN, "check.path.space", "fix.path.move", p))
    if system == "Windows" and len(s) > WINDOWS_PATH_MAX:
        out.append(Check("path_long", Status.WARN, "check.path.long", "fix.path.move", p))
    if not out:
        out.append(Check("path", Status.OK, "check.path.ok", params=p))
    return out


# ---------------------------------------------------------------- software checks (after install)


def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, str(getattr(mod, "__version__", ""))
    except Exception as e:  # ImportError, OSError (missing DLL), RuntimeError...
        log.debug("import %s failed: %r", name, e)
        return False, f"{type(e).__name__}: {e}"


# Required CUDA extensions, as import names. nvdiffrec is deliberately absent (PLAN 2.6).
REQUIRED_EXTENSIONS = ("flex_gemm", "cumesh", "o_voxel", "nvdiffrast.torch")


def check_software() -> list[Check]:
    ok, info = _try_import("torch")
    if not ok:
        return [Check("torch", Status.FAIL, "check.torch.missing", "fix.reinstall", {"err": info})]
    import torch

    out = []
    p = {"torch": torch.__version__, "cuda": torch.version.cuda or "-"}
    if torch.cuda.is_available():
        out.append(Check("torch", Status.OK, "check.torch.ok", params=p))
    else:
        out.append(Check("torch", Status.FAIL, "check.torch.no_cuda", "fix.torch.no_cuda", p))

    missing = []
    for name in REQUIRED_EXTENSIONS:
        ok, info = _try_import(name)
        if not ok:
            missing.append(f"{name} ({info})")
    if missing:
        out.append(
            Check(
                "extensions",
                Status.FAIL,
                "check.ext.missing",
                "fix.reinstall",
                {"names": ", ".join(missing)},
            )
        )
    else:
        out.append(
            Check(
                "extensions",
                Status.OK,
                "check.ext.ok",
                params={"names": ", ".join(REQUIRED_EXTENSIONS)},
            )
        )

    from .attention import choose_backend

    backend = choose_backend()
    if backend.sparse is None:
        out.append(
            Check(
                "attention",
                Status.FAIL,
                "check.attn.none",
                "fix.reinstall",
                {"dense": backend.dense},
            )
        )
    else:
        out.append(
            Check(
                "attention",
                Status.OK,
                "check.attn.ok",
                params={"dense": backend.dense, "sparse": backend.sparse},
            )
        )

    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    out.append(Check("allocator", Status.OK, "check.alloc.info", params={"conf": alloc or "-"}))
    return out


# ---------------------------------------------------------------- entry points


def run_hardware_checks(install_dir: Path, models_dir: Path) -> list[Check]:
    system = platform.system()
    checks = [check_os(system)]
    checks += check_gpu(probe_gpus())
    checks.append(check_ram(probe_ram_gib()))
    checks.append(check_disk(probe_disk_free_gib(models_dir), models_dir))
    checks += check_install_path(install_dir, system)
    return checks


def os_name() -> str:
    """platform.release() says "10" on Windows 11; the build number tells them apart."""
    system, release, version = platform.system(), platform.release(), platform.version()
    if system == "Windows" and release == "10":
        try:
            if int(version.split(".")[2]) >= 22000:
                release = "11"
        except (IndexError, ValueError):
            pass
    return f"{system} {release} ({version})"


def system_summary() -> dict[str, Any]:
    gpus = probe_gpus() or []
    return {
        "os": os_name(),
        "python": sys.version.split()[0],
        "gpus": [
            {
                "name": g.name,
                "vram_gib": round(g.vram_gib, 2),
                "compute_capability": f"{g.compute_capability[0]}.{g.compute_capability[1]}",
                "driver": g.driver_version,
            }
            for g in gpus
        ],
        "ram_gib": round(probe_ram_gib(), 1),
    }


def worst(checks: list[Check]) -> Status:
    order = [Status.FAIL, Status.WARN, Status.OK, Status.SKIP]
    for s in order:
        if any(c.status == s for c in checks):
            return s
    return Status.OK
