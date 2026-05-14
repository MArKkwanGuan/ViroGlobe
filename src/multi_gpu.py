from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


try:
    import cupy as cp
    from cupy.cuda import nccl as _nccl
    NCCL_AVAILABLE = True
except ImportError:
    NCCL_AVAILABLE = False
    cp = None
    _nccl = None


@dataclass(frozen=True)
class GPUDevice:
    global_rank: int
    local_rank: int
    device_id: int
    host_name: str


@dataclass
class DeviceMesh:
    devices: list[GPUDevice]
    n_gpus: int
    grid_rows: int
    grid_cols: int
    use_nccl: bool = field(default=False)

    @staticmethod
    def discover_available_gpus() -> list[int]:
        if cp is None:
            return []
        available = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if available:
            return [int(x) for x in available.split(",") if x.strip().isdigit()]
        device_count = cp.cuda.runtime.getDeviceCount()
        return list(range(device_count))

    @staticmethod
    def get_host_name() -> str:
        return os.environ.get("HOSTNAME", os.environ.get("SLURMD_NODENAME", "localhost"))

    @classmethod
    def create(
        cls,
        gpu_ids: list[int] | None = None,
        grid_rows: int = 1,
        grid_cols: int | None = None,
        use_nccl: bool = False,
    ) -> DeviceMesh:
        if gpu_ids is None:
            gpu_ids = cls.discover_available_gpus()

        n_gpus = len(gpu_ids)

        if grid_cols is None:
            grid_cols = n_gpus // grid_rows
            if grid_rows * grid_cols != n_gpus:
                grid_cols = n_gpus
                grid_rows = 1

        if grid_rows * grid_cols != n_gpus:
            raise ValueError(f"grid_rows ({grid_rows}) * grid_cols ({grid_cols}) must equal n_gpus ({n_gpus})")

        host_name = cls.get_host_name()
        devices = [
            GPUDevice(
                global_rank=i,
                local_rank=i,
                device_id=gpu_ids[i],
                host_name=host_name,
            )
            for i in range(n_gpus)
        ]

        return cls(
            devices=devices,
            n_gpus=n_gpus,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            use_nccl=use_nccl,
        )


@dataclass
class NCCLCommunicator:
    rank: int
    n_gpus: int
    stream: Any
    use_nccl: bool = field(default=False)
    _comm: Any = field(default=None, init=False, repr=False)

    def _get_comm(self):
        if self._comm is not None:
            return self._comm
        if not self.use_nccl or _nccl is None:
            return None
        try:
            self._comm = _nccl.NcclCommunicator(self.n_gpus, 0, 0)
            return self._comm
        except Exception:
            return None

    @classmethod
    def create(cls, rank: int, n_gpus: int, use_nccl: bool = False) -> NCCLCommunicator | None:
        if not use_nccl or cp is None:
            return None

        cp.cuda.Device(rank).use()
        stream = cp.cuda.Stream(non_blocking=True)
        return cls(rank=rank, n_gpus=n_gpus, stream=stream, use_nccl=True)

    def allreduce_sum(self, data: cp.ndarray) -> cp.ndarray:
        if not self.use_nccl or self.n_gpus <= 1:
            return data

        if data.size == 0:
            return data

        data = data.astype(cp.float32)

        comm = self._get_comm()
        if comm is None:
            local_sum = cp.zeros_like(data)
            local_sum[:] = data
            return local_sum

        nelems = data.size
        send_ptr = data.data.ptr
        recv_ptr = data.data.ptr

        comm.allReduce(
            send_ptr, recv_ptr, nelems,
            _nccl.NCCL_FLOAT32, _nccl.NCCL_SUM,
            self.stream.ptr
        )
        self.stream.synchronize()
        return data

    def barrier(self) -> None:
        if not self.use_nccl or self.n_gpus <= 1:
            return
        comm = self._get_comm()
        if comm is None:
            return
        comm.barrier(self.stream.ptr)
        self.stream.synchronize()

    def allgather(self, data: cp.ndarray) -> cp.ndarray:
        if not self.use_nccl or self.n_gpus <= 1:
            return data

        comm = self._get_comm()
        if comm is None:
            return data

        result = cp.zeros((self.n_gpus, data.size), dtype=data.dtype)
        comm.allGather(
            data.data.ptr, result.data.ptr, data.size,
            _nccl.NCCL_FLOAT32, self.stream.ptr
        )
        self.stream.synchronize()
        return result


class HaloExchange:
    def __init__(
        self,
        neighbor_ranks: dict[str, int | None],
        rank: int,
        stream: Any = None,
    ):
        self.neighbor_ranks = neighbor_ranks
        self.rank = rank
        self.stream = stream or cp.cuda.Stream()
        self.send_buffers: dict[str, cp.ndarray] = {}
        self.recv_buffers: dict[str, cp.ndarray] = {}
        self.pending_ops: list[tuple] = []

    def register_halo(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: Any = np.float32,
    ) -> None:
        if cp is None:
            return
        for direction in ["north", "south", "east", "west"]:
            key = f"{name}_{direction}"
            self.send_buffers[key] = cp.zeros(shape, dtype=dtype)
            self.recv_buffers[key] = cp.zeros(shape, dtype=dtype)

    def _cuda_peer_access(self, src_rank: int, dst_rank: int) -> bool:
        if cp is None:
            return False
        try:
            return cp.cuda.runtime.deviceCanAccessPeer(src_rank, dst_rank)
        except Exception:
            return False

    def exchange(
        self,
        name: str,
        local_data: cp.ndarray,
        direction: str,
        halo_width: int = 1,
    ) -> cp.ndarray:
        if cp is None:
            return local_data

        neighbor_rank = self.neighbor_ranks.get(direction)
        if neighbor_rank is None:
            return local_data

        key = f"{name}_{direction}"
        if key not in self.send_buffers:
            return local_data

        send_buf = self.send_buffers[key]
        recv_buf = self.recv_buffers[key]

        if send_buf.size == 0 or recv_buf.size == 0:
            return local_data

        send_buf.set(local_data)

        if self._cuda_peer_access(self.rank, neighbor_rank):
            self._p2p_transfer_pinned(send_buf, recv_buf, neighbor_rank)
        else:
            self._cpu_transfer(send_buf, recv_buf)

        self.stream.synchronize()
        return recv_buf.copy()

    def _p2p_transfer_pinned(
        self,
        send_buf: cp.ndarray,
        recv_buf: cp.ndarray,
        neighbor_rank: int,
    ) -> None:
        try:
            cp.cuda.runtime.deviceEnablePeerAccess(neighbor_rank)
        except Exception:
            pass

        with self.stream:
            cp.cuda.runtime.memcpyAsync(
                recv_buf.data.ptr,
                send_buf.data.ptr,
                send_buf.nbytes,
                cp.cuda.runtime.memcpyDeviceToDevice,
                self.stream.ptr
            )

    def _cpu_transfer(
        self,
        send_buf: cp.ndarray,
        recv_buf: cp.ndarray,
    ) -> None:
        tmp = cp.asnumpy(send_buf)
        recv_buf.set(tmp)

    def all_gather_halos(
        self,
        name: str,
        local_halo: cp.ndarray,
        directions: list[str],
    ) -> dict[str, cp.ndarray]:
        results = {}
        for direction in directions:
            if self.neighbor_ranks.get(direction) is not None:
                results[direction] = self.exchange(name, local_halo, direction)
            else:
                results[direction] = cp.zeros_like(local_halo)
        return results


class SpatialDecomposition:
    def __init__(
        self,
        n_gpus: int,
        grid_rows: int,
        grid_cols: int,
        total_lats: int,
        total_lons: int,
        active_mask_2d: np.ndarray | None = None,
    ):
        self.n_gpus = n_gpus
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.total_lats = total_lats
        self.total_lons = total_lons
        self.active_mask_2d = active_mask_2d

        self.lat_start: list[int] = []
        self.lat_end: list[int] = []
        self.lon_start: list[int] = []
        self.lon_end: list[int] = []

        self._compute_partition()

    @staticmethod
    def _equal_ranges(length: int, parts: int) -> list[tuple[int, int]]:
        if parts <= 0:
            raise ValueError("parts must be >= 1")
        base = length // parts
        remainder = length % parts
        ranges: list[tuple[int, int]] = []
        start = 0
        for part in range(parts):
            stop = start + base + (1 if part < remainder else 0)
            ranges.append((start, min(stop, length)))
            start = stop
        return ranges

    @classmethod
    def _weighted_ranges(cls, weights: np.ndarray, parts: int) -> list[tuple[int, int]]:
        length = int(weights.size)
        if parts <= 0:
            raise ValueError("parts must be >= 1")
        if length == 0:
            return [(0, 0) for _ in range(parts)]
        if parts >= length:
            return cls._equal_ranges(length, parts)

        weights = np.asarray(weights, dtype=np.float64)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            return cls._equal_ranges(length, parts)

        cumulative = np.cumsum(weights)
        boundaries = [0]
        for part in range(1, parts):
            remaining_parts = parts - part
            target = total_weight * part / parts
            boundary = int(np.searchsorted(cumulative, target, side="right"))
            boundary = max(boundary, boundaries[-1] + 1)
            boundary = min(boundary, length - remaining_parts)
            boundaries.append(boundary)
        boundaries.append(length)
        return [(boundaries[i], boundaries[i + 1]) for i in range(parts)]

    def _compute_partition(self) -> None:
        if self.active_mask_2d is None:
            row_ranges = self._equal_ranges(self.total_lats, self.grid_rows)
            col_ranges_by_row = [self._equal_ranges(self.total_lons, self.grid_cols) for _ in row_ranges]
        else:
            if self.active_mask_2d.shape != (self.total_lats, self.total_lons):
                raise ValueError(
                    f"active_mask_2d shape {self.active_mask_2d.shape} does not match "
                    f"({self.total_lats}, {self.total_lons})"
                )
            row_weights = np.asarray(self.active_mask_2d.sum(axis=1), dtype=np.float64)
            row_ranges = self._weighted_ranges(row_weights, self.grid_rows)
            col_ranges_by_row = []
            for lat_s, lat_e in row_ranges:
                row_mask = self.active_mask_2d[lat_s:lat_e]
                col_weights = np.asarray(row_mask.sum(axis=0), dtype=np.float64)
                col_ranges_by_row.append(self._weighted_ranges(col_weights, self.grid_cols))

        for row, (lat_s, lat_e) in enumerate(row_ranges):
            for lon_s, lon_e in col_ranges_by_row[row]:
                self.lat_start.append(lat_s)
                self.lat_end.append(lat_e)
                self.lon_start.append(lon_s)
                self.lon_end.append(lon_e)

    def get_partition(self, rank: int) -> tuple[int, int, int, int]:
        return (
            self.lat_start[rank],
            self.lat_end[rank],
            self.lon_start[rank],
            self.lon_end[rank],
        )

    def get_neighbor_ranks(self, rank: int) -> dict[str, int | None]:
        row = rank // self.grid_cols
        col = rank % self.grid_cols

        neighbors: dict[str, int | None] = {
            "north": None,
            "south": None,
            "east": None,
            "west": None,
        }

        if row > 0:
            neighbors["north"] = (row - 1) * self.grid_cols + col
        if row < self.grid_rows - 1:
            neighbors["south"] = (row + 1) * self.grid_cols + col
        if col > 0:
            neighbors["west"] = row * self.grid_cols + (col - 1)
        if col < self.grid_cols - 1:
            neighbors["east"] = row * self.grid_cols + (col + 1)

        return neighbors

    def get_halo_sizes(self, rank: int, halo_width: int = 1) -> dict[str, tuple[int, int]]:
        lat_s, lat_e, lon_s, lon_e = self.get_partition(rank)
        lat_h = lat_e - lat_s
        lon_h = lon_e - lon_s

        return {
            "north": (halo_width, lon_h),
            "south": (halo_width, lon_h),
            "east": (lat_h, halo_width),
            "west": (lat_h, halo_width),
        }

    def get_local_extent(self, rank: int) -> tuple[int, int]:
        lat_s, lat_e, lon_s, lon_e = self.get_partition(rank)
        return lat_e - lat_s, lon_e - lon_s

    def get_global_extent(self) -> tuple[int, int]:
        return self.total_lats, self.total_lons
