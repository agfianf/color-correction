from enum import Enum

from pydantic import BaseModel, Field


class GPUType(str, Enum):
    NVIDIA = "NVIDIA"
    AMD = "AMD"
    APPLE = "Apple Integrated"
    UNKNOWN = "Unknown GPU"


class CPUArchitecture(str, Enum):
    INTEL = "Intel"
    AMD = "AMD"
    ARM = "ARM"
    APPLE = "Apple Silicon"
    UNKNOWN = "Unknown"


class DeviceSpecs(BaseModel):
    """
    Device specifications schema.

    Attributes
    ----------
    os_name : str
        Operating system name.
    cpu_arch : CPUArchitecture
        CPU architecture.
    gpu_type : GPUType
        GPU type.
    is_apple_silicon : bool
        Whether the device is Apple Silicon.
    """

    os_name: str = Field(..., description="Operating system name")
    cpu_arch: CPUArchitecture = Field(
        CPUArchitecture.UNKNOWN,
        description="CPU architecture",
    )
    gpu_type: GPUType = Field(GPUType.UNKNOWN, description="GPU type")
    is_apple_silicon: bool = Field(False, description="Whether device is Apple Silicon")
