"""
JAX configuration for TurboDiff.

Automatically configures JAX for different hardware platforms:
- Apple Silicon (M1/M2/M3): CPU 
- Intel CPU: CPU
- NVIDIA GPU: GPU with CUDA
- AMD GPU: CPU (JAX GPU support limited)

Override with environment variables:
- TURBODIFF_JAX_PLATFORM: 'cpu', 'gpu', or 'tpu'
- TURBODIFF_JAX_X64: '1' to enable 64-bit precision
- TURBODIFF_JAX_DISABLE_JIT: '1' to disable JIT compilation
"""

import os
import platform
import warnings
from jax import config as jax_config


def _detect_platform():
    """
    Detect the best JAX platform for the current hardware.

    Returns:
        str: 'cpu', 'gpu', or 'tpu'
    """
    system = platform.system()
    machine = platform.machine().lower()

    # Check for explicit override first
    override = os.getenv("TURBODIFF_JAX_PLATFORM", "").lower()
    if override in ("cpu", "gpu", "tpu"):
        return override

    # macOS
    if system == "Darwin":
        if "arm" in machine or machine == "arm64":
            # Apple Silicon - Force CPU
            return "cpu"
        else:
            # Intel Mac - CPU only
            return "cpu"

    # Linux
    elif system == "Linux":
        # Check if CUDA is available
        try:
            # Temporarily set to check what's available
            os.environ["JAX_PLATFORMS"] = ""

            # Import jax to trigger device discovery
            from jax import devices

            # Check for CUDA/GPU devices
            try:
                gpu_devices = devices("gpu")
                if len(gpu_devices) > 0:
                    return "gpu"
            except Exception:
                pass

            # Check for TPU devices
            try:
                tpu_devices = devices("tpu")
                if len(tpu_devices) > 0:
                    return "tpu"
            except Exception:
                pass

        except Exception:
            pass

        # Fallback: Check nvidia-smi
        import subprocess

        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=2)
            if result.returncode == 0:
                return "gpu"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Default to CPU
        return "cpu"

    # Windows
    elif system == "Windows":
        # JAX GPU support on Windows is limited
        # Check for NVIDIA GPU via nvidia-smi
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=2, shell=True
            )
            if result.returncode == 0:
                # GPU available, but warn about limited support
                warnings.warn(
                    "GPU detected on Windows. JAX GPU support on Windows is experimental.",
                    UserWarning,
                )
                return "gpu"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return "cpu"

    # Unknown system
    else:
        warnings.warn(f"Unknown system: {system}. Defaulting to CPU.", UserWarning)
        return "cpu"


def _configure_platform():
    """Configure JAX platform before importing JAX."""
    detected_platform = _detect_platform()

    # Set JAX_PLATFORMS environment variable BEFORE importing JAX
    os.environ["JAX_PLATFORMS"] = detected_platform

    return detected_platform


def _print_config_info(platform_name, x64_enabled, jit_disabled):
    """Print configuration information."""
    system = platform.system()
    machine = platform.machine()

    print("[TurboDiff JAX Config]")
    print(f"  System: {system} ({machine})")
    print(f"  Platform: {platform_name.upper()}")
    print(f"  Precision: {'64-bit' if x64_enabled else '32-bit'}")
    print(f"  JIT: {'Disabled' if jit_disabled else 'Enabled'}")

    if platform_name == "cpu" and system == "Darwin" and "arm" in machine.lower():
        print(
            "  Note: Apple Silicon detected - using CPU (Metal backend is experimental)"
        )


# ============================================================================
# Main Configuration
# ============================================================================

# detect and set platform
platform_name = _configure_platform()

x64_enabled = os.getenv("TURBODIFF_JAX_X64", "0") == "1"
jax_config.update("jax_enable_x64", x64_enabled)

# explicitly set platform name
jax_config.update("jax_platform_name", platform_name)

# configure JIT
jit_disabled = os.getenv("TURBODIFF_JAX_DISABLE_JIT", "0") == "1"
if jit_disabled:
    jax_config.update("jax_disable_jit", True)

# print configuration info (can be silenced with TURBODIFF_SILENT=1)
if os.getenv("TURBODIFF_SILENT", "0") != "1":
    _print_config_info(platform_name, x64_enabled, jit_disabled)
