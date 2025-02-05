import platform
import subprocess
from typing import Any

from color_correction.schemas.device import (
    CPUArchitecture,
    DeviceSpecs,
    GPUType,
)


def detect_darwin(specs: dict[str, Any]) -> dict[str, Any]:
    """Detect hardware specifications for macOS (Darwin).

    Parameters
    ----------
    specs : dict
        Base specifications dictionary with OS information.

    Returns
    -------
    dict
        Updated specifications with CPU and GPU information.

    Notes
    -----
    Detects:
    1. Apple Silicon vs Intel via sysctl
    2. GPU type (Apple/AMD/NVIDIA) via system_profiler
    """
    try:
        cpu_info = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
        ).decode()
        specs["is_apple_silicon"] = "Apple" in cpu_info
        specs["cpu_arch"] = (
            CPUArchitecture.APPLE if "Apple" in cpu_info else CPUArchitecture.INTEL
        )
    except subprocess.SubprocessError:
        specs["cpu_arch"] = CPUArchitecture.UNKNOWN

    try:
        gpu_info = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
        ).decode()
        if "Apple M" in gpu_info:
            specs["gpu_type"] = GPUType.APPLE
        elif "AMD" in gpu_info:
            specs["gpu_type"] = GPUType.AMD
        elif "NVIDIA" in gpu_info:
            specs["gpu_type"] = GPUType.NVIDIA
        else:
            specs["gpu_type"] = GPUType.UNKNOWN
    except subprocess.SubprocessError:
        specs["gpu_type"] = GPUType.UNKNOWN

    return specs


def detect_linux(specs: dict[str, Any]) -> dict[str, Any]:
    """Detect hardware specifications for Linux.

    Parameters
    ----------
    specs : dict
        Base specifications dictionary with OS information.

    Returns
    -------
    dict
        Updated specifications with CPU and GPU information.

    Notes
    -----
    Detects:
    1. CPU architecture (ARM/x86_64) via lscpu
    2. GPU type (NVIDIA/AMD) via nvidia-smi or lspci
    """
    try:
        cpu_info = subprocess.check_output("lscpu", shell=True).decode().lower()
        specs["cpu_arch"] = (
            CPUArchitecture.ARM if "aarch64" in cpu_info else CPUArchitecture.INTEL
        )
    except subprocess.SubprocessError:
        specs["cpu_arch"] = CPUArchitecture(platform.machine())

    # GPU detection
    try:
        subprocess.check_output("nvidia-smi", shell=True)
        specs["gpu_type"] = GPUType.NVIDIA
    except subprocess.SubprocessError:
        try:
            amd_info = (
                subprocess.check_output(
                    "lspci | grep -i amd",
                    shell=True,
                )
                .decode()
                .lower()
            )
            specs["gpu_type"] = (
                GPUType.AMD
                if ("vga" in amd_info or "amd" in amd_info)
                else GPUType.UNKNOWN
            )
        except subprocess.SubprocessError:
            specs["gpu_type"] = GPUType.UNKNOWN

    return specs


def detect_windows(specs: dict[str, Any]) -> dict[str, Any]:
    """Detect hardware specifications for Windows.

    Parameters
    ----------
    specs : dict
        Base specifications dictionary with OS information.

    Returns
    -------
    dict
        Updated specifications with CPU and GPU information.
    """
    proc = platform.processor().lower()
    if "intel" in proc:
        specs["cpu_arch"] = CPUArchitecture.INTEL
    elif "amd" in proc:
        specs["cpu_arch"] = CPUArchitecture.AMD
    elif "arm" in proc:
        specs["cpu_arch"] = CPUArchitecture.ARM
    else:
        specs["cpu_arch"] = CPUArchitecture.UNKNOWN

    specs["gpu_type"] = GPUType.UNKNOWN
    return specs


def get_device_specs() -> DeviceSpecs:
    """Get device hardware specifications.

    Returns
    -------
    DeviceSpecs
        Pydantic model containing device specifications.
    """
    specs = {
        "os_name": platform.system(),
        "cpu_arch": CPUArchitecture.UNKNOWN,
        "gpu_type": GPUType.UNKNOWN,
        "is_apple_silicon": False,
    }

    detector_map = {
        "Darwin": detect_darwin,
        "Linux": detect_linux,
        "Windows": detect_windows,
    }

    detect_func = detector_map.get(specs["os_name"], lambda x: x)
    specs = detect_func(specs)

    return DeviceSpecs(**specs)


if __name__ == "__main__":
    specs = get_device_specs()
    print(f"Detected specs: {specs.model_dump()}")
