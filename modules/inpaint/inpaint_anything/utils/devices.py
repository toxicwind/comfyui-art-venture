import sys
import torch
from packaging import version


# before torch version 1.13, has_mps is only available in nightly pytorch and macOS 12.3+,
# use check `getattr` and try it for compatibility.
# in torch version 1.13, backends.mps.is_available() and backends.mps.is_built() are introduced in to check mps availabilty,
# since torch 2.0.1+ nightly build, getattr(torch, 'has_mps', False) was deprecated, see https://github.com/pytorch/pytorch/pull/103279
def has_mps() -> bool:
    if version.parse(torch.__version__) <= version.parse("2.0.1"):
        if not getattr(torch, 'has_mps', False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False
    else:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def torch_mps_gc() -> None:
    try:
        from torch.mps import empty_cache
        empty_cache()
    except Exception:
        print("MPS garbage collection failed")


def get_optimal_device_name():
    if torch.cuda.is_available():
        return "cuda"

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        torch_mps_gc()