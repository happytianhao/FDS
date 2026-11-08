"""Read-only viewer server with batched original images and MP4 byte ranges."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

IMAGE_BATCH_PATH = "/__harness_viewer__/images"
MAX_IMAGE_PATHS = 64
MAX_REQUEST_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_BATCH_BYTES = 32 * 1024 * 1024
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def image_batch(directory, paths):
    """Return original image bytes with offsets; no decoding, resizing or filesystem writes."""
    if not isinstance(paths, list) or not 1 <= len(paths) <= MAX_IMAGE_PATHS:
        raise ValueError("Expected 1–64 image paths")
    root = Path(directory).resolve()
    files = []
    for url in paths:
        if not isinstance(url, str) or len(url) > 4096:
            raise ValueError("Invalid image path")
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
            raise ValueError("Only server-relative image paths are accepted")
        path = (root / unquote(parsed.path).lstrip("/")).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in IMAGE_TYPES:
            raise ValueError("Image path is outside the served directory or is not an image")
        files.append((url, path))
    entries, chunks, offset = [], [], 0
    for url, path in files:
        try:
            with path.open("rb") as stream:
                payload = stream.read(MAX_IMAGE_BYTES + 1)
            if len(payload) > MAX_IMAGE_BYTES or offset + len(payload) > MAX_BATCH_BYTES:
                raise ValueError("Image batch exceeds byte limit")
            status = 200
        except (FileNotFoundError, IsADirectoryError):
            payload, status = b"", 404
        except PermissionError:
            payload, status = b"", 403
        entries.append(
            {
                "path": url,
                "status": status,
                "type": IMAGE_TYPES[path.suffix.lower()],
                "offset": offset,
                "size": len(payload),
            }
        )
        chunks.append(payload)
        offset += len(payload)
    header = json.dumps({"version": 1, "images": entries}, separators=(",", ":")).encode()
    return len(header).to_bytes(4, "big") + header + b"".join(chunks)


class ViewerHandler(SimpleHTTPRequestHandler):
    remaining = None

    def do_POST(self):
        if urlsplit(self.path).path != IMAGE_BATCH_PATH:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES or self.headers.get("Transfer-Encoding"):
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or set(request) != {"paths"}:
                raise ValueError("Expected image path list")
            payload = image_batch(self.directory, request["paths"])
        except (ValueError, OSError) as exc:
            self.send_error(400, str(exc))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-fds-images")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "private, no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        if urlsplit(self.path).path.endswith((".html", "/")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        self.remaining = None
        requested = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not requested or not path.is_file():
            return super().send_head()
        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        stat = path.stat()
        modified = self.date_time_string(stat.st_mtime)
        if self.headers.get("If-Range") not in (None, modified):
            stream.close()
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
        start = end = None
        if match and any(match.groups()) and stat.st_size:
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), stat.st_size - 1) if right else stat.st_size - 1
            elif int(right) > 0:
                start, end = max(0, stat.st_size - int(right)), stat.st_size - 1
        if start is None or not 0 <= start <= end < stat.st_size:
            stream.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{stat.st_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
        self.send_header("Last-Modified", modified)
        self.end_headers()
        stream.seek(start)
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                shutil.copyfileobj(source, outputfile)
            else:
                remaining = self.remaining
                while remaining > 0:
                    block = source.read(min(64 * 1024, remaining))
                    if not block:
                        break
                    outputfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel older range requests while seeking.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=5011)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error("--directory must be an existing directory")
    handler = partial(ViewerHandler, directory=str(directory))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(
            f"Serving {directory} at http://{args.bind}:{args.port} (video byte ranges enabled)",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
