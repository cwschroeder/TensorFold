"""Fail unless the pinned patch base is still the upstream main tip."""
import subprocess
import sys
EXPECTED = "1f938b1b57132e173da458b2e8880c9184a071e2"
URL = "https://github.com/ashhart/TensorFold.git"
actual = subprocess.check_output(["git", "ls-remote", URL, "refs/heads/main"], text=True, timeout=60).split()[0]
if actual != EXPECTED:
    sys.exit(f"Upstream moved: pinned={EXPECTED}, current={actual}; rebase and retest before building or promoting")
if subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True).returncode == 0:
    subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED, "HEAD"], check=True)
print(f"upstream_verified={actual}")
