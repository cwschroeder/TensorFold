"""Sample the owned Mac process footprint and system swap; retain failed probes rather than inventing a peak."""

import ctypes
import json
import subprocess
import threading
import time


class Monitor:
    """Write process-bound samples from launch until that child has exited."""

    def __init__(self, server):
        self.server = server
        self.done = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.guarded_run, daemon=True)
        self.thread.start()

    def guarded_run(self):
        """Surface setup and file errors on join instead of silently losing the sampler."""
        try:
            self.run()
        except BaseException as error:
            self.error = error

    def run(self):
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        began = time.monotonic()
        with (self.server.out / "memory.jsonl").open("a") as file:
            while not self.done.is_set():
                row = {"instance": self.server.instance, "phase": self.server.stage,
                       "pid": self.server.process.pid, "seconds": time.monotonic() - began}
                info = (ctypes.c_uint64 * 43)()
                try:
                    if lib.proc_pid_rusage(self.server.process.pid, 4, ctypes.byref(info)) != 0:
                        raise OSError("process footprint probe failed")
                    row["physical_footprint_bytes"] = info[9]
                    row["swap"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=5).strip()
                except (OSError, subprocess.SubprocessError) as error:
                    row["error"] = type(error).__name__ + ": " + str(error)
                file.write(json.dumps(row) + "\n")
                file.flush()
                self.done.wait(5)

    def stop(self):
        """Join the sampler before completing cleanup evidence."""
        self.done.set()
        self.thread.join(10)
        if self.thread.is_alive():
            raise RuntimeError("owned memory sampler did not stop")
        if self.error is not None:
            raise RuntimeError("owned memory sampler failed") from self.error
