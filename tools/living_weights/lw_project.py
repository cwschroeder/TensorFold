"""Measure fictional DE/EN project facts, duplicates and updates on a disposable checkpoint and a real restart."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import signal
import shutil
import socket
import statistics
import subprocess
import time
import tempfile

from lw_project_io import ask, get, teach
from lw_project_memory import Monitor
from lw_project_score import load, score

HERE = Path(__file__).resolve().parent


def write(path, value):
    """Atomically retain intermediate evidence without replacing a completed run."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


@contextmanager
def protected_cleanup():
    """Finish owned cleanup even if another terminal signal arrives."""
    saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


@contextmanager
def owned_spawn():
    """Defer terminal interruption until the caller has assigned the spawned child."""
    pending = []
    saved = {sig: signal.signal(sig, lambda number, frame: pending.append(number))
             for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    if pending:
        raise KeyboardInterrupt("interrupted during owned spawn")


@contextmanager
def host_lock(receipt, path=None):
    """Exclude overlapping runners; retain the lock if child exit cannot be confirmed."""
    path = path or Path(tempfile.gettempdir()) / f"tensorfold-project-learning-{os.getuid()}.lock"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(str(os.getpid()))
    try:
        yield
    finally:
        with protected_cleanup():
            if receipt.get("owned_server_stopped", True):
                path.unlink()


def digest(path):
    """Hash a file without reading the checkpoint into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def weights(model):
    """Pin all original weight files and config, rejecting learned or incomplete folders."""
    config = json.loads((model / "config.json").read_text())
    quant = config.get("quantization", {})
    if (config.get("model_type") != "nemotron_h" or config.get("tensorfold_learned_neurons")
            or quant.get("bits") != 4 or list(model.glob("living-weights*.safetensors"))):
        raise ValueError("requires an untaught Nemotron checkpoint")
    names = set(json.loads((model / "model.safetensors.index.json").read_text())["weight_map"].values())
    if not names or any(Path(n).name != n or not (model / n).is_file() for n in names):
        raise ValueError("incomplete or invalid shard index")
    return {n: digest(model / n) for n in sorted(names | {"config.json"})}


def preflight(model, output):
    """Require an idle Apple Silicon host and room for checkpoint writes before starting any child."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("this managed experiment is qualified for Apple Silicon only")
    pressure = subprocess.check_output(["memory_pressure", "-Q"], text=True)
    import re
    free = re.search(r"free percentage:\s*(\d+)%", pressure)
    if not free or int(free[1]) < 40:
        raise RuntimeError("insufficient memory headroom")
    inventory = subprocess.check_output(["ps", "-axo", "pid=,comm=,args="], text=True)
    conflicts = []
    for row in inventory.splitlines():
        pid, executable, arguments = row.split(maxsplit=2)
        name = Path(executable).name.lower()
        native = name in ("tensorfold", "tensorfold-native", "omlx", "ollama")
        python_model = name.startswith("python") and re.search(r"(?:-m\s+mlx_(?:lm|vlm)\.server|[/ ](?:omlx|tensorfold)(?: |$))", arguments)
        if (native or python_model) and int(pid) not in (os.getpid(), os.getppid()):
            conflicts.append(int(pid))
    if conflicts:
        raise RuntimeError("competing model process ids: " + str(conflicts))
    size = sum(p.stat().st_size for p in model.glob("*.safetensors"))
    free_disk = shutil.disk_usage(output.parent).free
    if free_disk < 2 * size + 10 * 1024**3:
        raise RuntimeError("insufficient disk headroom")
    return {"memory_free_percentage": int(free[1]), "disk_free_bytes": free_disk,
            "ram_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))}


class Server:
    """Own one local native child and an authentication key kept out of the public receipt."""

    def __init__(self, binary, model, out, learning):
        self.out, self.learning = out, learning
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.key = out / ("learning.key" if learning else "restart.key")
        token = secrets.token_urlsafe(32)
        with open(self.key, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as file:
            file.write(token)
        self.headers = {"Authorization": "Bearer " + token}
        args = [str(binary), "serve", str(model), "--host", "127.0.0.1", "--port", str(port),
                "--name", "project-learning-test", "--parallel", "1", "--context", "4096",
                "--temperature", "0", "--no-thinking", "--prompt-cache-gib", "0",
                "--api-key-file", str(self.key), "--no-update-check"]
        if learning:
            args += ["--slide", "--slide-graph", str(out / "graph.json")]
        self.args = args
        self.log = None
        self.process = None
        self.monitor = None
        self.instance, self.stage = secrets.token_hex(16), "loading"

    def start(self):
        """Launch after the caller owns this object, including readiness failures."""
        self.log = (self.out / ("learning.log" if self.learning else "restart.log")).open("w")
        with owned_spawn():
            self.process = subprocess.Popen(self.args, stdout=self.log, stderr=self.log)
        self.monitor = Monitor(self)
        self.ready()
        self.stage = "idle"

    def ready(self):
        """Tie readiness to this child and the intended model name, not just an open port."""
        until = time.monotonic() + 300
        while time.monotonic() < until:
            if self.process.poll() is not None:
                raise RuntimeError("owned server exited before readiness")
            try:
                data = get(self.url, "/v1/models", self.headers)
                if any(m["id"] == "project-learning-test" for m in data["data"]):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise TimeoutError("owned model readiness timed out")

    def close(self):
        """Stop only this child; do not continue the restart test until it has exited."""
        with protected_cleanup():
            try:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=10)
            finally:
                try:
                    if self.log is not None:
                        self.log.close()
                finally:
                    if self.monitor is not None:
                        self.monitor.stop()


def probe(server, spec, state, label, out):
    """Retain every visible reply, reasoning, token usage and format verdict in a fresh chat."""
    server.stage = label
    result = {"phase": label, "instance": server.instance, "learning_enabled": server.learning,
              "hashes": spec["hashes"], "state": dict(state), "rows": [], "complete": False}
    for question in spec["questions"]:
        for thinking in (False, True):
            reply = ask(server.url, question["text"], thinking, server.headers)
            reply.update(question=question["id"], kind=question["kind"], language=question["language"],
                         thinking=thinking, score=score(question, reply["answer"], reply["finish"], spec["cases"], state))
            result["rows"].append(reply)
            write(out / (label + ".json"), result)
    result["complete"] = True
    write(out / (label + ".json"), result)
    print(label, sum(r["score"]["correct"] for r in result["rows"]), "/", len(result["rows"]), flush=True)
    return result


def summarise(phases, lessons):
    """Keep recall, controls, neighbour changes and time separate; never turn a rejected lesson into transport failure."""
    summary = {"lessons": [{"id": s["id"], "status": s["status"],
                            "seconds_including_save": s["seconds_including_save"]} for s in lessons], "phases": []}
    baseline = {(r["question"], r["thinking"]): r for r in phases[0]["rows"]}
    for phase in phases:
        row = {"phase": phase["phase"], "groups": {}}
        for kind in ("recall", "neighbour", "control"):
            selected = [r for r in phase["rows"] if r["kind"] == kind]
            row["groups"][kind] = {"correct": sum(r["score"]["correct"] for r in selected),
                                   "total": len(selected), "byte_changes": sum(
                                       r["answer"] != baseline[(r["question"], r["thinking"])]["answer"]
                                       for r in selected)}
        row["median_request_seconds"] = statistics.median(r["seconds"] for r in phase["rows"])
        summary["phases"].append(row)
    return summary


def execute(binary, model, out, spec, receipt):
    """Teach only corpus lesson texts, probe after each write, then restart without the learner."""
    server, phases, lessons, state = None, [], [], {}
    copy = out / "checkpoint"
    if platform.system() == "Darwin":
        subprocess.run(["cp", "-cRL", str(model), str(copy)], check=True)
    else:
        shutil.copytree(model, copy, symlinks=False)
    try:
        server = Server(binary, copy, out, True)
        receipt["owned_server_stopped"] = False
        server.start()
        phases.append(probe(server, spec, state, "baseline", out))
        for step in spec["steps"]:
            record = {"id": step["id"], "kind": step["kind"], "hashes": spec["hashes"],
                      "text_sha256": hashlib.sha256(step["text"].encode()).hexdigest()}
            print("learning", step["id"], flush=True)
            server.stage = step["id"] + "-learning"
            teach(server.url, step, record, lambda r: write(out / (step["id"] + "-learn.json"), r), server.headers)
            lessons.append(record)
            state[step["fact"]] = step["version"]
            phases.append(probe(server, spec, state, step["id"], out))
        previous = server.instance
        server.close()
        server = Server(binary, copy, out, False)
        server.start()
        assert previous != server.instance
        phases.append(probe(server, spec, state, "restart", out))
        receipt.update(status="completed", restart_verified=True, summary=summarise(phases, lessons))
    finally:
        try:
            if server is not None:
                server.close()
        finally:
            receipt["owned_server_stopped"] = server is None or server.process is None or server.process.poll() is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--lessons", type=Path, default=HERE / "project_lessons.json")
    parser.add_argument("--questions", type=Path, default=HERE / "project_questions.json")
    args = parser.parse_args()
    spec = load(args.lessons, args.questions)
    build = json.loads(args.build_receipt.read_text())
    binary_sha = digest(args.binary)
    if build["binary_sha256"] != binary_sha or not build.get("upstream_commit") or not build.get("compiler_version"):
        raise ValueError("missing or mismatched build pins")
    receipt = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with host_lock(receipt):
        resources = preflight(args.model, args.output)
        args.output.mkdir(parents=True, exist_ok=False)
        receipt.update(schema=1, status="running", hashes=spec["hashes"], binary_sha256=binary_sha,
                       build=build, resources=resources, source_before=weights(args.model))
        write(args.output / "receipt.json", receipt)
        try:
            execute(args.binary.resolve(), args.model.resolve(), args.output, spec, receipt)
        except BaseException as error:
            receipt.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            with protected_cleanup():
                receipt["source_after"] = weights(args.model)
                receipt["source_unchanged"] = receipt["source_after"] == receipt["source_before"]
                write(args.output / "receipt.json", receipt)
        if not receipt["source_unchanged"]:
            raise RuntimeError("source checkpoint changed")


def interrupted(_signal, _frame):
    """Convert termination into an exception so the owned child is stopped in finally."""
    raise KeyboardInterrupt("interrupted")


if __name__ == "__main__":
    main()
