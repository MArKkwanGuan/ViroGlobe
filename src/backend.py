from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


try:
    import cupy as cp
    from cupy.cuda import nccl as _nccl
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    _nccl = None


@dataclass(frozen=True)
class Backend:
    xp: Any
    name: str
    gpu_enabled: bool

    def asnumpy(self, arr: Any) -> np.ndarray:
        if self.gpu_enabled:
            return self.xp.asnumpy(arr)
        return np.asarray(arr)


@dataclass
class MultiGPUBackend:
    backends: list[Backend]
    device_mesh: Any
    rank: int
    n_gpus: int
    use_nccl: bool
    device_id: int
    _nccl_comm: Any = field(default=None, init=False, repr=False)
    _pending_ops: list = field(default_factory=list, init=False, repr=False)
    _comm_stream: Any = field(default=None, init=False, repr=False)

    @property
    def gpu_enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"multi_gpu_{self.n_gpus}gpu"

    @property
    def xp(self) -> Any:
        return self.backends[self.rank].xp if self.rank < len(self.backends) else self.backends[0].xp

    @property
    def current_backend(self) -> Backend:
        return self.backends[self.rank] if self.rank < len(self.backends) else self.backends[0]

    def activate_device(self) -> None:
        if cp is not None:
            cp.cuda.Device(self.device_id).use()

    def _get_comm_stream(self) -> Any:
        if self._comm_stream is not None:
            return self._comm_stream
        if cp is None:
            return None
        self._comm_stream = cp.cuda.Stream(non_blocking=True)
        return self._comm_stream

    def _get_nccl_comm(self):
        if not self.use_nccl:
            return None
        if self._nccl_comm is not None:
            return self._nccl_comm
        if _nccl is None:
            return None
        try:
            self._nccl_comm = _nccl.NcclCommunicator(self.n_gpus, 0, 0)
            return self._nccl_comm
        except Exception:
            return None

    def asnumpy(self, arr: Any) -> np.ndarray:
        if hasattr(arr, 'get'):
            return arr.get()
        if hasattr(arr, 'asnumpy'):
            return arr.asnumpy()
        return np.asarray(arr)

    def allreduce_sum(self, data: cp.ndarray) -> cp.ndarray:
        if not self.use_nccl or self.n_gpus <= 1:
            return data

        if data.size == 0:
            return data

        data = data.astype(cp.float32)
        comm = self._get_nccl_comm()

        if comm is None:
            return data

        nelems = data.size
        send_ptr = data.data.ptr
        recv_ptr = data.data.ptr

        stream = self._get_comm_stream()
        if stream is None:
            return data

        try:
            comm.allReduce(
                send_ptr, recv_ptr, nelems,
                _nccl.NCCL_FLOAT32, _nccl.NCCL_SUM,
                stream.ptr
            )
            stream.synchronize()
        except Exception:
            pass

        return data

    def allreduce_sum_async(self, data: cp.ndarray, stream: Any = None) -> cp.ndarray:
        if not self.use_nccl or self.n_gpus <= 1:
            return data

        if data.size == 0:
            return data

        data = data.astype(cp.float32)
        comm = self._get_nccl_comm()

        if comm is None:
            return data

        nelems = data.size
        send_ptr = data.data.ptr
        recv_ptr = data.data.ptr

        target_stream = stream or self._get_comm_stream()
        if target_stream is None:
            return data

        try:
            comm.allReduce(
                send_ptr, recv_ptr, nelems,
                _nccl.NCCL_FLOAT32, _nccl.NCCL_SUM,
                target_stream.ptr
            )
        except Exception:
            pass

        return data

    def allreduce_sum_batch(self, data_list: list[cp.ndarray]) -> list[cp.ndarray]:
        if not self.use_nccl or self.n_gpus <= 1:
            return data_list

        if not data_list:
            return data_list

        comm = self._get_nccl_comm()
        stream = self._get_comm_stream()

        if comm is None or stream is None:
            return [self.allreduce_sum(d) for d in data_list]

        results = []
        flattened_data = []
        shapes = []
        sizes = []

        for data in data_list:
            if data.size == 0:
                results.append(data)
                continue
            flat = data.astype(cp.float32).ravel()
            shapes.append(data.shape)
            sizes.append(data.size)
            flattened_data.append(flat)

        if not flattened_data:
            return results

        total_size = sum(sizes)
        concat_data = cp.concatenate(flattened_data) if len(flattened_data) > 1 else flattened_data[0]

        try:
            nelems = concat_data.size
            send_ptr = concat_data.data.ptr
            recv_ptr = concat_data.data.ptr

            comm.allReduce(
                send_ptr, recv_ptr, nelems,
                _nccl.NCCL_FLOAT32, _nccl.NCCL_SUM,
                stream.ptr
            )
            stream.synchronize()

            offset = 0
            for i, data in enumerate(data_list):
                if data.size == 0:
                    continue
                size = sizes[i]
                result_flat = concat_data[offset:offset + size]
                results.append(result_flat.reshape(shapes[i]))
                offset += size
        except Exception:
            results = [self.allreduce_sum(d) for d in data_list]

        return results

    def barrier(self) -> None:
        if not self.use_nccl or self.n_gpus <= 1:
            return
        comm = self._get_nccl_comm()
        if comm is None:
            return
        stream = self._get_comm_stream()
        if stream is None:
            return
        try:
            comm.barrier(stream.ptr)
            stream.synchronize()
        except Exception:
            pass

    def barrier_async(self, stream: Any = None) -> None:
        if not self.use_nccl or self.n_gpus <= 1:
            return
        comm = self._get_nccl_comm()
        if comm is None:
            return
        target_stream = stream or self._get_comm_stream()
        if target_stream is None:
            return
        try:
            comm.barrier(target_stream.ptr)
        except Exception:
            pass

    def synchronize(self) -> None:
        stream = self._get_comm_stream()
        if stream is not None:
            stream.synchronize()

    def asnumpy_on_rank(self, arr: Any, target_rank: int) -> np.ndarray:
        if target_rank == self.rank:
            return self.asnumpy(arr)
        return np.asarray(arr)


def get_backend(force_gpu: bool | None = None, require_gpu: bool = False) -> Backend:
    use_gpu = os.environ.get("USE_GPU", "1") == "1" if force_gpu is None else force_gpu
    if use_gpu:
        try:
            import cupy as cp
            return Backend(xp=cp, name="cupy", gpu_enabled=True)
        except ImportError:
            if require_gpu:
                raise RuntimeError("GPU backend requested, but CuPy is not available in the current environment.")
    return Backend(xp=np, name="numpy", gpu_enabled=False)


def get_multi_gpu_backend(
    gpu_ids: list[int] | None = None,
    grid_rows: int = 1,
    grid_cols: int | None = None,
    force_gpu: bool = True,
) -> MultiGPUBackend | Backend:
    if not force_gpu:
        return get_backend(force_gpu=False)

    if not CUPY_AVAILABLE:
        raise RuntimeError("Multi-GPU execution requires CuPy, but it is not available in the current environment.")

    try:
        from .multi_gpu import DeviceMesh

        mesh = DeviceMesh.create(
            gpu_ids=gpu_ids,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            use_nccl=_nccl is not None,
        )
        n_gpus = mesh.n_gpus

        if n_gpus <= 1:
            raise RuntimeError("Multi-GPU execution requires at least 2 visible GPUs.")

        rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = min(rank, n_gpus - 1)
        device_id = mesh.devices[rank].device_id
        cp.cuda.Device(device_id).use()

        backends = [Backend(xp=cp, name=f"cupy_gpu_{device.device_id}", gpu_enabled=True) for device in mesh.devices]

        return MultiGPUBackend(
            backends=backends,
            device_mesh=mesh,
            rank=rank,
            n_gpus=n_gpus,
            use_nccl=mesh.use_nccl,
            device_id=device_id,
        )
    except Exception as e:
        raise RuntimeError("Failed to initialize the multi-GPU backend.") from e


def is_distributed_run() -> bool:
    return "WORLD_SIZE" in os.environ and "RANK" in os.environ


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_global_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))
