"""Small, dependency-light LeRobotDataset v3.0 writer and validator.

The public schema follows LeRobot v3.0.  Low-dimensional samples are written to
Parquet, RGB observations to MP4, and episode boundaries to relational metadata.
The implementation intentionally avoids torch and the full lerobot package so it
can run in the isolated ``pushenv`` environment.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODE_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _dtype(name: str) -> np.dtype:
    aliases = {"float32": np.float32, "float64": np.float64, "int64": np.int64,
               "int32": np.int32, "bool": np.bool_, "uint8": np.uint8}
    if name not in aliases:
        raise ValueError(f"Unsupported LeRobot dtype {name!r}.")
    return np.dtype(aliases[name])


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class LeRobotWriter:
    """Write episodes to a local LeRobotDataset v3.0 repository."""

    def __init__(self, root: str | Path, schema: dict[str, Any], task: str,
                 provenance: dict[str, Any] | None = None, max_frames_per_file: int = 10_000):
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Output dataset is not empty: {self.root}")
        if not task.strip():
            raise ValueError("task must contain a natural-language instruction.")
        if max_frames_per_file < 1:
            raise ValueError("max_frames_per_file must be positive.")
        self.schema = copy.deepcopy(schema)
        self.task = task.strip()
        self.provenance = copy.deepcopy(provenance or {})
        self.fps = int(self.schema["fps"])
        if self.fps <= 0 or self.fps != self.schema["fps"]:
            raise ValueError("fps must be a positive integer.")
        self.features = copy.deepcopy(self.schema["features"])
        overlap = set(self.features) & set(DEFAULT_FEATURES)
        if overlap:
            raise ValueError(f"Schema must not redefine generated features: {sorted(overlap)}")
        self.features.update(copy.deepcopy(DEFAULT_FEATURES))
        for spec in self.features.values():
            spec["fps"] = float(self.fps)
            if spec["dtype"] == "video":
                height, width, channels = spec["shape"]
                spec["info"] = {
                    "video.height": height,
                    "video.width": width,
                    "video.channels": channels,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": float(self.fps),
                    "has_audio": False,
                    "is_depth_map": False,
                }
        self.video_keys = [key for key, spec in self.features.items() if spec["dtype"] == "video"]
        self.data_keys = [key for key in self.features if key not in self.video_keys]
        if "observation.state" not in self.features or "action" not in self.features:
            raise ValueError("LeRobot datasets require observation.state and action features.")
        self.max_frames_per_file = int(max_frames_per_file)
        self.root.mkdir(parents=True, exist_ok=True)
        self._episodes: list[dict[str, Any]] = []
        self._pending: list[dict[str, np.ndarray]] = []
        self._pending_frames = 0
        self._total_frames = 0
        self._file_index = 0
        self._stats: dict[str, dict[str, Any]] = {}
        self._closed = False
        self.aborted = False
        self._checkpoint_file_index = 0
        self._metadata_checkpoint: dict[Path, bytes] = {}

    def __enter__(self) -> "LeRobotWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _validate_episode(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        expected = set(self.features) - set(DEFAULT_FEATURES)
        if set(data) != expected:
            raise ValueError(f"Episode features differ from schema: {sorted(set(data) ^ expected)}")
        result: dict[str, np.ndarray] = {}
        lengths = set()
        for key in expected:
            spec = self.features[key]
            value = np.asarray(data[key])
            shape = tuple(spec["shape"])
            if value.ndim < 1 or value.shape[1:] != shape:
                raise ValueError(f"{key} must have shape [frames, {', '.join(map(str, shape))}], got {value.shape}.")
            lengths.add(len(value))
            if spec["dtype"] == "video":
                if value.dtype != np.uint8:
                    raise ValueError(f"{key} video frames must be uint8 RGB.")
            else:
                value = value.astype(_dtype(spec["dtype"]), copy=False)
                if value.dtype.kind == "f" and not np.isfinite(value).all():
                    raise ValueError(f"{key} contains non-finite values.")
            result[key] = value
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("Every episode feature must have the same positive frame count.")
        return result

    def write_episode(self, data: dict[str, np.ndarray]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write after close().")
        episode = self._validate_episode(data)
        length = len(next(iter(episode.values())))
        if self._pending and self._pending_frames + length > self.max_frames_per_file:
            self._flush_file()
        self._pending.append(episode)
        self._pending_frames += length
        if self._pending_frames >= self.max_frames_per_file:
            self._flush_file()

    def _moments(self, key: str, value: np.ndarray):
        """Reduce RGB exactly by pixel counts, with bounded scratch memory."""
        if self.features[key]["dtype"] == "video":
            channels = value.shape[-1]
            histogram = np.zeros((channels, 256), dtype=np.int64)
            batch_size = max(1, (1024 * 1024) // int(np.prod(value.shape[1:])))
            for start in range(0, len(value), batch_size):
                pixels = value[start:start + batch_size].reshape(-1, channels)
                for channel in range(channels):
                    histogram[channel] += np.bincount(pixels[:, channel], minlength=256)
            levels = np.arange(256, dtype=np.float64) / 255.0
            count = int(histogram[0].sum())
            total = (histogram * levels).sum(axis=1)
            total_sq = (histogram * np.square(levels)).sum(axis=1)
            occupied = histogram > 0
            minimum = np.argmax(occupied, axis=1) / 255.0
            maximum = (255 - np.argmax(occupied[:, ::-1], axis=1)) / 255.0
            return (channels, 1, 1), count, total, total_sq, minimum, maximum
        return self._float_moments(key, value)

    def _float_moments(self, key: str, value: np.ndarray):
        """Keep the original float reduction for low-dimensional vectors."""
        flat = value.reshape(len(value), -1).astype(np.float64)
        return ((flat.shape[1],), len(flat), flat.sum(axis=0), np.square(flat).sum(axis=0),
                flat.min(axis=0), flat.max(axis=0))

    def _episode_stats(self, key: str, value: np.ndarray, moments=None) -> dict[str, list]:
        output_shape, count, total, total_sq, minimum, maximum = (
            self._moments(key, value) if moments is None else moments
        )
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0))
        if self.features[key]["dtype"] == "bool":
            minimum, maximum = minimum.astype(bool), maximum.astype(bool)
        return {
            "min": minimum.reshape(output_shape).tolist(),
            "max": maximum.reshape(output_shape).tolist(),
            "mean": mean.reshape(output_shape).tolist(),
            "std": std.reshape(output_shape).tolist(),
            "count": [len(value)],
        }

    def _update_stats(self, key: str, value: np.ndarray, moments=None) -> None:
        output_shape, moment_count, total, total_sq, minimum, maximum = (
            self._moments(key, value) if moments is None else moments
        )
        current = self._stats.get(key)
        if current is None:
            self._stats[key] = {"count": len(value), "moment_count": moment_count,
                                "output_shape": output_shape, "sum": total, "sum_sq": total_sq,
                                "min": minimum, "max": maximum,
                                "is_bool": self.features[key]["dtype"] == "bool"}
        else:
            current["count"] += len(value)
            current["moment_count"] += moment_count
            current["sum"] += total
            current["sum_sq"] += total_sq
            current["min"] = np.minimum(current["min"], minimum)
            current["max"] = np.maximum(current["max"], maximum)

    def _flush_file(self) -> None:
        if not self._pending:
            return
        file_index = self._file_index
        start_global = self._total_frames
        rows: dict[str, list[np.ndarray]] = {key: [] for key in self.data_keys}
        local_offset = 0
        for local_episode, episode in enumerate(self._pending):
            length = len(next(iter(episode.values())))
            episode_index = len(self._episodes)
            # Episode and dataset statistics use the same reductions. Cache
            # only channel/feature summaries, never another RGB frame batch.
            episode_moments = {}
            generated = {
                "timestamp": np.arange(length, dtype=np.float32) / self.fps,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(self._total_frames, self._total_frames + length, dtype=np.int64),
                "task_index": np.zeros(length, dtype=np.int64),
            }
            for key in self.data_keys:
                value = generated[key] if key in generated else episode[key]
                rows[key].append(value)
                moments = self._moments(key, value)
                episode_moments[key] = moments
                self._update_stats(key, value, moments)
            record: dict[str, Any] = {
                "episode_index": episode_index,
                "tasks": [self.task],
                "length": length,
                "dataset_from_index": self._total_frames,
                "dataset_to_index": self._total_frames + length,
                "data/chunk_index": 0,
                "data/file_index": file_index,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for key in self.video_keys:
                record[f"videos/{key}/chunk_index"] = 0
                record[f"videos/{key}/file_index"] = file_index
                record[f"videos/{key}/from_timestamp"] = local_offset / self.fps
                record[f"videos/{key}/to_timestamp"] = (local_offset + length) / self.fps
            for key in self.features:
                value = generated[key] if key in generated else episode[key]
                moments = episode_moments[key] if key in episode_moments else self._moments(key, value)
                for statistic, statistic_value in self._episode_stats(key, value, moments).items():
                    record[f"stats/{key}/{statistic}"] = statistic_value
                if key in self.video_keys:
                    self._update_stats(key, value, moments)
            self._episodes.append(record)
            local_offset += length
            self._total_frames += length

        columns: dict[str, Any] = {}
        for key, pieces in rows.items():
            value = np.concatenate(pieces, axis=0)
            shape = self.features[key]["shape"]
            columns[key] = value.reshape(-1) if shape == [1] else value.tolist()
        path = self.root / DATA_PATH.format(chunk_index=0, file_index=file_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(columns), path, compression="zstd")

        for key in self.video_keys:
            video_path = self.root / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=file_index)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=self.fps, codec="libx264", pixelformat="yuv420p",
                                        quality=9, macro_block_size=None, ffmpeg_log_level="error")
            try:
                for episode in self._pending:
                    for frame in episode[key]:
                        writer.append_data(frame)
            finally:
                writer.close()
        self._pending.clear()
        self._pending_frames = 0
        self._file_index += 1
        if self._total_frames <= start_global:
            raise RuntimeError("Writer made no progress while flushing.")

    def checkpoint(self) -> None:
        """Persist complete episodes and metadata while keeping recording open."""
        if self._closed:
            return
        self._flush_file()
        if not self._episodes:
            raise ValueError("Cannot finalize an empty dataset.")
        episode_path = self.root / EPISODE_PATH.format(chunk_index=0, file_index=0)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(self._episodes), episode_path, compression="zstd")
        task_frame = pd.DataFrame({"task_index": [0]}, index=pd.Index([self.task]))
        task_frame.to_parquet(self.root / "meta/tasks.parquet")

        stats: dict[str, Any] = {}
        for key, values in self._stats.items():
            count = values["moment_count"]
            shape = values["output_shape"]
            mean = values["sum"] / count
            variance = np.maximum(values["sum_sq"] / count - np.square(mean), 0)
            minimum, maximum = values["min"], values["max"]
            if values["is_bool"]:
                minimum, maximum = minimum.astype(bool), maximum.astype(bool)
            stats[key] = {"min": minimum.reshape(shape).tolist(), "max": maximum.reshape(shape).tolist(),
                          "mean": mean.reshape(shape).tolist(), "std": np.sqrt(variance).reshape(shape).tolist(),
                          "count": [values["count"]]}
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.schema.get("robot_type", self.schema.get("environment", "unknown")),
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 500,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "features": self.features,
        }
        _json(self.root / "meta/info.json", info)
        _json(self.root / "meta/stats.json", stats)
        _json(self.root / "meta/pushenv_schema.json", self.schema)
        _json(self.root / "meta/provenance.json", self.provenance)
        self._metadata_checkpoint = {
            path: path.read_bytes() for path in self._metadata_paths()
        }
        self._checkpoint_file_index = self._file_index

    def _metadata_paths(self) -> list[Path]:
        return [self.root / EPISODE_PATH.format(chunk_index=0, file_index=0),
                *[self.root / "meta" / name for name in
                  ("tasks.parquet", "info.json", "stats.json", "pushenv_schema.json", "provenance.json")]]

    def abort(self) -> None:
        """Release pending image buffers and preserve the last saved checkpoint."""
        if self.aborted:
            return
        self.aborted = self._closed = True
        self._pending.clear()
        self._pending_frames = 0
        for index in range(self._checkpoint_file_index, self._file_index + 1):
            paths = [self.root / DATA_PATH.format(chunk_index=0, file_index=index)]
            paths.extend(self.root / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=index)
                         for key in self.video_keys)
            for path in paths:
                path.unlink(missing_ok=True)
        for path in self._metadata_paths():
            if path in self._metadata_checkpoint:
                path.write_bytes(self._metadata_checkpoint[path])
            else:
                path.unlink(missing_ok=True)

    def close(self) -> None:
        if not self._closed:
            try:
                self.checkpoint()
            except BaseException:
                self.abort()
                raise
            self._closed = True


def validate_dataset(root: str | Path) -> dict[str, Any]:
    """Verify cross-file row counts, required columns and episode boundaries."""
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    data_files = sorted((root / "data").rglob("*.parquet"))
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not data_files or not episode_files:
        raise ValueError("Dataset is missing Parquet data or episode metadata.")
    required = set(DEFAULT_FEATURES) | {"observation.state", "action"}
    required.update(key for key, feature in info["features"].items() if feature["dtype"] != "video")
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    rows = 0
    last_index = -1
    for path in data_files:
        table = pq.read_table(path)
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        indices = table["index"].to_numpy()
        if len(indices) and (indices[0] != last_index + 1 or np.any(np.diff(indices) != 1)):
            raise ValueError("Global indices are not contiguous.")
        if len(indices):
            last_index = int(indices[-1])
        rows += len(table)
    episodes = pa.concat_tables([pq.read_table(path) for path in episode_files]).to_pylist()
    if rows != info["total_frames"] or len(episodes) != info["total_episodes"]:
        raise ValueError("Metadata totals differ from stored rows.")
    if episodes[0]["dataset_from_index"] != 0 or episodes[-1]["dataset_to_index"] != rows:
        raise ValueError("Episode ranges do not cover the dataset.")
    if any(a["dataset_to_index"] != b["dataset_from_index"] for a, b in zip(episodes, episodes[1:])):
        raise ValueError("Episode ranges contain a gap or overlap.")
    video_ends: dict[tuple[str, Path], float] = {}
    for episode in episodes:
        for key in video_keys:
            path = root / info["video_path"].format(
                video_key=key, chunk_index=episode[f"videos/{key}/chunk_index"],
                file_index=episode[f"videos/{key}/file_index"])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty video for {key}: {path}")
            start, end = (episode[f"videos/{key}/{name}_timestamp"] for name in ("from", "to"))
            if not np.isfinite([start, end]).all() or start < 0 or not np.isclose(
                end - start, episode["length"] / info["fps"], atol=1e-4):
                raise ValueError(f"Video timestamps disagree with episode length: {key}")
            group = (key, path)
            if not np.isclose(start, video_ends.get(group, 0.0), atol=1e-4):
                raise ValueError(f"Video episode ranges contain a gap or overlap: {key}")
            video_ends[group] = end
    for (key, path), end in video_ends.items():
        with imageio.get_reader(path) as reader:
            metadata = reader.get_meta_data()
            height, width, _ = info["features"][key]["shape"]
            if metadata["size"] != (width, height) or not np.isclose(metadata["fps"], info["fps"], atol=1e-3):
                raise ValueError(f"Video dimensions or FPS differ from the feature schema: {key}")
            if reader.count_frames() != round(end * info["fps"]):
                raise ValueError(f"Video frame count disagrees with episode ranges: {key}")
    task_frame = pd.read_parquet(root / "meta/tasks.parquet")
    if list(task_frame.columns) != ["task_index"] or len(task_frame) != info["total_tasks"]:
        raise ValueError("tasks.parquet has an invalid schema.")
    return {"root": str(root.resolve()), "frames": rows, "episodes": len(episodes),
            "tasks": len(task_frame), "data_files": len(data_files),
            "video_files": len(list((root / "videos").rglob("*.mp4")))}

