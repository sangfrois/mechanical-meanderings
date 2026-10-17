"""StreamDiffusionTD GPU identity and process-local launch configuration.

Embed as gpu_runtime under the StreamDiffusionTD COMP. No global environment
changes, DLL replacement, GPU-index assumptions, or runtime restart. The physical
GPU is obtained from a TD-owned TOP allocation, not the current CUDA context
(which another extension may temporarily have switched).
"""
import ctypes
import os
from pathlib import Path
import uuid

_identity = None


def library_path(owner):
    base = Path(str(owner.par.Basefolder.eval()))
    for env_name in ('venv', '.venv'):
        candidate = base / env_name / 'Lib' / 'site-packages'
        if (candidate / 'cuda_link' / '__init__.py').is_file():
            return str(candidate)
    raise RuntimeError('cuda-link is missing from StreamDiffusion Basefolder; check Install, not the system CUDA DLLs')


def _ok(status, operation):
    if status:
        raise RuntimeError('{} failed (CUDA driver status {})'.format(operation, status))


def gpu_identity(owner, refresh=False):
    global _identity
    if _identity is not None and not refresh:
        return dict(_identity)
    source = owner.op('fit_')
    if source is None:
        raise RuntimeError('StreamDiffusion input TOP fit_ is missing')
    driver = ctypes.WinDLL('nvcuda.dll')
    driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    driver.cuPointerGetAttribute.restype = ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
    driver.cuDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    driver.cuDeviceGetUuid.restype = ctypes.c_int
    # Keep this TD allocation alive until the pointer query completes.
    memory = source.cudaMemory()
    try:
        device = ctypes.c_int()
        _ok(driver.cuPointerGetAttribute(ctypes.byref(device), 9, int(memory.ptr)), 'TD pointer GPU query')
        bus = ctypes.create_string_buffer(32)
        _ok(driver.cuDeviceGetPCIBusId(bus, 32, device.value), 'GPU PCI query')
        uid = (ctypes.c_ubyte * 16)()
        _ok(driver.cuDeviceGetUuid(ctypes.byref(uid), device.value), 'GPU UUID query')
        identity = dict(ordinal=device.value, uuid='GPU-' + str(uuid.UUID(bytes=bytes(uid))),
            pci_bus=bus.value.decode('ascii'))
    finally:
        del memory
    _identity = identity
    return dict(identity)


def child_environment(owner):
    """Only the new SD process gets a visibility mask. Never print this mapping."""
    identity = gpu_identity(owner, refresh=True)
    # Inherit credentials opaquely, as normal subprocess launch does; no logging.
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = identity['uuid']
    environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    environment['CUDALINK_LIB_PATH'] = library_path(owner)
    return environment
