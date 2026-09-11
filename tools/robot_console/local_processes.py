"""Recover local workspace drivers without relying on a previous window's QProcess."""
from dataclasses import dataclass
from pathlib import Path
import os
import signal


@dataclass(frozen=True)
class LocalProcess:
    pid: int
    started: str


def discover(kind: str, repo: Path, proc: Path = Path('/proc')) -> list[LocalProcess]:
    """Match this user's exact workspace executable and ROS domain, never a name substring."""
    executable = str(repo / ('install/spinerobot_tracking/lib/spinerobot_tracking/ndi_tracker'
                             if kind == 'ndi' else 'install/xarm_api/lib/xarm_api/xarm_driver_node'))
    found = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            args = (directory / 'cmdline').read_bytes().split(b'\0')
            argv = [a.decode(errors='surrogateescape') for a in args if a]
            # NDI is a Python script; xArm is a native executable.
            matches = bool(argv) and (argv[0] == executable or
                       (kind == 'ndi' and len(argv) > 1 and argv[1] == executable))
            if not matches:
                continue
            env = (directory / 'environ').read_bytes().split(b'\0')
            if b'ROS_DOMAIN_ID=42' not in env:
                continue
            stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            found.append(LocalProcess(int(directory.name), stat[19]))
        except (OSError, IndexError, ValueError):
            continue
    return found


def stop(process: LocalProcess, kind: str, repo: Path) -> bool:
    """Pin the PID before rechecking identity to avoid signalling a reused PID."""
    try:
        fd = os.pidfd_open(process.pid)
    except ProcessLookupError:
        return False
    try:
        if process not in discover(kind, repo):
            return False
        signal.pidfd_send_signal(fd, signal.SIGINT)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)
