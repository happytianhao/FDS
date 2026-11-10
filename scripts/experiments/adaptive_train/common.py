"""Durable state and Linux event primitives for the bounded train queue."""
import contextlib
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import psutil


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def lock(path, nonblocking=False):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def identity_alive(pid, created, command=None):
    if not isinstance(pid, int) or pid <= 0 or not isinstance(created, (int, float)):
        return False
    try:
        proc = psutil.Process(pid)
        return (proc.create_time() == created and proc.status() != psutil.STATUS_ZOMBIE
                and (command is None or proc.cmdline() == command))
    except psutil.Error:
        return False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def process_exit_fd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


class FileEvents:
    """inotify wakeups; no timed polling and no content in event payloads."""
    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1')
        self.paths = set()
        self.filters = {}

    def watch(self, path, names=None):
        path = str(path)
        if path not in self.paths:
            wd = self.libc.inotify_add_watch(self.fd, os.fsencode(path), 0x100 | 0x80 | 0x8)
            if wd < 0:
                raise OSError(ctypes.get_errno(), 'inotify_add_watch')
            self.paths.add(path)
            self.filters[wd] = None if names is None else set(names)

    def drain(self):
        relevant = False
        try:
            while data := os.read(self.fd, 65536):
                offset = 0
                while offset < len(data):
                    wd, mask, _, size = struct.unpack_from('iIII', data, offset)
                    name = os.fsdecode(data[offset + 16:offset + 16 + size].split(b'\0', 1)[0])
                    names = self.filters.get(wd)
                    relevant |= bool(mask & 0x4000) or names is None or name in names
                    offset += 16 + size
        except BlockingIOError:
            pass
        return relevant

    def close(self):
        os.close(self.fd)
