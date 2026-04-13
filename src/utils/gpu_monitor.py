"""GPU monitoring utilities using pynvml."""

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import pynvml


def init_nvml():
    """Initialize NVML. Call once at program start."""
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def shutdown_nvml():
    """Shutdown NVML. Call at program end."""
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def get_gpu_stats(device_index: int = 0) -> dict:
    """Get GPU stats for a single device.

    Returns dict with keys: mem_used_gb, mem_total_gb, mem_pct, util_pct, temp_c
    """
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "mem_used_gb": mem.used / 1e9,
            "mem_total_gb": mem.total / 1e9,
            "mem_pct": 100 * mem.used / mem.total,
            "util_pct": util.gpu,
            "temp_c": temp,
        }
    except Exception:
        return {
            "mem_used_gb": 0, "mem_total_gb": 0, "mem_pct": 0,
            "util_pct": 0, "temp_c": 0,
        }


def get_all_gpu_stats(num_gpus: int = 4) -> list[dict]:
    """Get stats for all GPUs."""
    return [get_gpu_stats(i) for i in range(num_gpus)]


def format_gpu_stats(device_index: int = 0) -> str:
    """Format GPU stats as a compact string."""
    s = get_gpu_stats(device_index)
    return (f"GPU{device_index}: {s['mem_used_gb']:.1f}/{s['mem_total_gb']:.1f}GB "
            f"| {s['util_pct']}% util | {s['temp_c']}°C")


def format_all_gpu_stats(num_gpus: int = 4) -> str:
    """Format all GPU stats as a single line."""
    parts = []
    for i in range(num_gpus):
        s = get_gpu_stats(i)
        parts.append(f"GPU{i}:{s['mem_used_gb']:.0f}GB/{s['util_pct']}%/{s['temp_c']}°C")
    return " | ".join(parts)
