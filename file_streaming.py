import shlex
import subprocess
from subprocess import Popen
from typing import Callable, Optional


class DelayedRealtimeStream:
    def __init__(self, process_factory: Callable[[], subprocess.Popen]):
        self._process_factory = process_factory
        self._process: Optional[Popen] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.process.__exit__(exc_type, value, traceback)

    def get_return_code(self) -> int:
        return self.process.returncode

    @property
    def process(self):
        if self._process is None:
            self._process = self._process_factory()
        return self._process

    async def read(self, size: int):
        assert self.process.stdout is not None
        return self.process.stdout.read(size)


def get_audio_stream_proc(filepath):
    """
    Use ffmpeg to stream audio file chunks in real-time
    """
    ffmpeg_cmd = f"ffmpeg -re -i {filepath} -ar 48000 -ac 1 -f s16le -acodec pcm_s16le pipe:1"
    ffmpeg_proc = subprocess.Popen(shlex.split(ffmpeg_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return ffmpeg_proc
