"""Build current-source Metal kernels and portable M3 native code; refuse a stale base."""
import pathlib
import subprocess
import sys
p = pathlib.Path(__file__).resolve().parent
subprocess.run([sys.executable, str(p / "check_upstream.py")], check=True)
zig = pathlib.Path(sys.argv[1]).resolve()
assert subprocess.check_output([str(zig), "version"], text=True).strip() == "0.17.0"
subprocess.run([str(zig), "build", "native", "test", "test-golden", "tf-core-lane-check", "-Dcpu=apple_m1", "--summary", "all"], check=True)
