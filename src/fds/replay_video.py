"""Cache a human-only MP4 from recorded RGB frames, without changing source evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import unquote

from PIL import Image, ImageDraw

VERSION = 1
HEADER = 26
FOOTER = 28


def _sources(frames, destination):
    """Stat all inputs so ordinary image edits, appends and reordered frames invalidate the cache."""
    paths, signature = {}, []
    for frame in frames:
        images = {}
        for camera, url in frame["images"].items():
            path = (destination / unquote(url)).resolve()
            stat = path.stat()
            paths[url] = path
            images[camera] = [url, stat.st_size, stat.st_mtime_ns]
        signature.append({k: frame.get(k) for k in ("frame", "step", "cue")} | {"images": images})
    return paths, signature


def _encode(frames, paths, cameras, fps, target, encoder):
    # Native image resolution is retained; smaller images are centered in black tiles.
    width = height = 2
    for path in set(paths.values()):
        with Image.open(path) as im:
            width, height = max(width, im.width), max(height, im.height)
    width += width % 2
    height += height % 2
    columns = min(3, len(cameras))
    rows = math.ceil(len(cameras) / columns)
    size = (width * columns, (height + HEADER) * rows + FOOTER)
    command = [
        encoder,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{size[0]}x{size[1]}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(max(1, round(fps))),
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(target),
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        timeout = threading.Timer(1800, process.kill)
        timeout.start()
        try:
            for frame in frames:
                canvas = Image.new("RGB", size, (14, 24, 37))
                draw = ImageDraw.Draw(canvas)
                for index, camera in enumerate(cameras):
                    x, y = (index % columns) * width, (index // columns) * (height + HEADER)
                    draw.text((x + 8, y + 6), camera, fill="white")
                    url = frame["images"].get(camera)
                    if url:
                        with Image.open(paths[url]) as im:
                            canvas.paste(
                                im.convert("RGB"),
                                (
                                    x + (width - im.width) // 2,
                                    y + HEADER + (height - im.height) // 2,
                                ),
                            )
                    else:
                        draw.text((x + 8, y + HEADER + 12), "Not recorded", fill="white")
                label = f"Frame {frame['frame']} | step {frame.get('step')} | {'DEMO' if frame.get('cue') else 'EXECUTION'}"
                draw.text((8, size[1] - FOOTER + 8), label, fill="white")
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Video encoder failed")
        except (OSError, RuntimeError) as exc:
            errors.seek(0)
            raise RuntimeError(f"{exc}: {errors.read().decode(errors='replace')[-1600:]}") from exc
        finally:
            timeout.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            try:
                process.stdin.close()
            except OSError:
                pass
    return size


def build_replay(episode, destination, frames):
    """One recorded frame per video frame, in exactly the viewer's order (no interpolation)."""
    if len(frames) < 2:
        return {"status": "unavailable", "reason": "至少需要两帧记录才能合成视频。"}
    episode, destination = Path(episode), Path(destination)
    try:
        metadata = json.loads((episode / "benchmark_metadata.json").read_text())
    except (OSError, ValueError):
        metadata = {}
    fps = metadata.get("control_freq", 10)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or not 1 <= fps <= 120
    ):
        fps = 10
    cameras = sorted(
        {c for f in frames for c in f["images"]},
        key=lambda c: ({"front": 0, "right": 1, "wrist": 2}.get(c, 3), c),
    )
    if not cameras:
        return {"status": "unavailable", "reason": "没有已记录的相机图片。"}
    directory = destination / "media"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{episode.parent.name}_{episode.name}"
    target, manifest = directory / f"{stem}.mp4", directory / f"{stem}.json"
    try:
        # A directory lock serializes concurrent refreshes without touching episode files.
        with (directory / ".encode.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            paths, signature = _sources(frames, destination)
            settings = {"version": VERSION, "fps": fps, "cameras": cameras, "frames": signature}
            fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            try:
                cached = json.loads(manifest.read_text())
                if (
                    cached["fingerprint"] == fingerprint
                    and target.stat().st_size == cached["bytes"]
                ):
                    return cached
            except (OSError, ValueError, KeyError):
                pass
            encoder = shutil.which("ffmpeg")
            if encoder is None:
                raise RuntimeError("缺少 ffmpeg；安装后重新生成可视化即可获得视频。")
            fd, temporary = tempfile.mkstemp(suffix=".mp4", prefix=".encoding-", dir=directory)
            os.close(fd)
            temporary = Path(temporary)
            try:
                size = _encode(frames, paths, cameras, fps, temporary, encoder)
                if _sources(frames, destination)[1] != signature:
                    raise RuntimeError("原始帧在合成期间发生变化，请重新生成视频。")
                if not temporary.stat().st_size:
                    raise RuntimeError("Video encoder produced an empty file")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            result = {
                "status": "ready",
                "fingerprint": fingerprint,
                "bytes": target.stat().st_size,
                "path": str(target.relative_to(destination)),
                "fps": fps,
                "frames": len(frames),
                "duration_seconds": len(frames) / fps,
                "cameras": cameras,
                "size": list(size),
                "timing": "one_recorded_frame_per_video_frame",
                "human_audit_only": True,
            }
            temporary_manifest = manifest.with_suffix(".json.tmp")
            temporary_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            temporary_manifest.replace(manifest)
            return result
    except (OSError, RuntimeError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
