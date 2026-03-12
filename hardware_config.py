#!/usr/bin/env python3

from dataclasses import dataclass


def _load_local_config():
    try:
        import local_config
    except ModuleNotFoundError:
        return None

    return local_config


def _local_config_value(local_config_module, *names, default):
    if local_config_module is None:
        return default

    for name in names:
        if hasattr(local_config_module, name):
            return getattr(local_config_module, name)

    return default


_LOCAL_CONFIG = _load_local_config()


@dataclass
class HardwareConfig:
    ip_address: str = _local_config_value(
        _LOCAL_CONFIG,
        "IP_ADDRESS",
        "ip_address",
        default="localhost",
    )
    port: int = _local_config_value(
        _LOCAL_CONFIG,
        "PORT",
        "port",
        default=11111,
    )
    fpga_clk_freq_MHz: float = _local_config_value(
        _LOCAL_CONFIG,
        "FPGA_CLK_FREQ_MHZ",
        "fpga_clk_freq_MHz",
        default=122.88,
    )
    grad_board: str = _local_config_value(
        _LOCAL_CONFIG,
        "GRAD_BOARD",
        "grad_board",
        default="gpa-fhdo",
    )
    gpa_fhdo_current_per_volt: float = _local_config_value(
        _LOCAL_CONFIG,
        "GPA_FHDO_CURRENT_PER_VOLT",
        "gpa_fhdo_current_per_volt",
        default=2.5,
    )
