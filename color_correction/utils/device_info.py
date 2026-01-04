import platform
import subprocess
from collections.abc import Callable
from typing import TypedDict

from color_correction.schemas.device import (
    CPUArchitecture,
    DeviceSpecs,
    GPUType,
)


class DeviceSpecsDict(TypedDict, total=False):
    """TypedDict for device specifications before validation."""

    os_name: str
    cpu_arch: CPUArchitecture
    gpu_type: GPUType
    is_apple_silicon: bool


def detect_darwin(specs: DeviceSpecsDict) -> DeviceSpecsDict:
    """
    Detect hardware specifications on macOS, including CPU and GPU details.

    Parameters
    ----------
    specs : DeviceSpecsDict
        Initial dictionary containing OS information.

    Returns
    -------
    DeviceSpecsDict
        Updated dictionary with CPU architecture and GPU type for macOS.
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


def detect_linux(specs: DeviceSpecsDict) -> DeviceSpecsDict:
    """
    Detect hardware specifications on Linux systems.

    Parameters
    ----------
    specs : DeviceSpecsDict
        Initial dictionary with OS information.

    Returns
    -------
    DeviceSpecsDict
        Updated dictionary with CPU architecture and GPU type for Linux.
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


def detect_windows(specs: DeviceSpecsDict) -> DeviceSpecsDict:
    """
    Detect hardware specifications on Windows systems.

    Parameters
    ----------
    specs : DeviceSpecsDict
        Initial dictionary with OS information.

    Returns
    -------
    DeviceSpecsDict
        Updated dictionary with CPU architecture and GPU type for Windows.
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
    """
    Retrieve a structured set of device hardware specifications.

    Returns
    -------
    DeviceSpecs
        An object containing OS name, CPU architecture, GPU type,
        and Apple Silicon flag.
    """
    specs: DeviceSpecsDict = {
        "os_name": platform.system(),
        "cpu_arch": CPUArchitecture.UNKNOWN,
        "gpu_type": GPUType.UNKNOWN,
        "is_apple_silicon": False,
    }

    detector_map: dict[str, Callable[[DeviceSpecsDict], DeviceSpecsDict]] = {
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
