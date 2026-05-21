import logging
import re
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        name_path = Path(name).expanduser()
        if name_path.exists():
            self.h5_path = name_path
        else:
            self.h5_path = Path(cache_dir or ".").expanduser() / f"{name}.h5"

        self.h5_file: h5py.File | None = None
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip
        self.transform = transform
        self._cache: dict[str, np.ndarray] = {}

        with h5py.File(self.h5_path, "r") as f:
            self.lengths = f["ep_len"][:]
            self.offsets = f["ep_offset"][:]
            self._keys = keys_to_load or [
                key for key in f.keys() if key not in ("ep_len", "ep_offset")
            ]

            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info("Cached '%s' from '%s'", key, self.h5_path)

        self.clip_indices = [
            (ep, start)
            for ep, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)
        if "action" in steps:
            steps["action"] = steps["action"].reshape(self.num_steps, -1)
        return steps

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start = self.offsets[ep_idx] + start
        g_end = self.offsets[ep_idx] + end
        steps = {}

        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]
            if col != "action":
                data = data[:: self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                val = data[0] if len(data) > 0 else b""
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        return {col: self.h5_file[col][row_idx] for col in self._keys}

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        self._open()

        if isinstance(source, str):
            source = [key for key in self.h5_file.keys() if re.match(source, key)]

        merged = np.concatenate([self._get_col(src) for src in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info("Merged columns %s into '%s' and cached it", source, target)

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1
