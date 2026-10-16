"""CPU CLI receipts distinguish a validated head from a plain or damaged checkpoint."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ROOT / "zig-out/bin/tf-clef-inspect"
FIXTURE = ROOT / "zig/tests/fixtures/clef"
CONFIG = {
    "model_type": "qwen3_5_text", "hidden_size": 16, "intermediate_size": 32,
    "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 8, "vocab_size": 97, "linear_num_key_heads": 2,
    "linear_num_value_heads": 4, "linear_key_head_dim": 4,
    "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4, "eos_token_id": 2,
}


class InspectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path / "config.json").write_text(json.dumps(CONFIG))

    def pair(self):
        for name in ("joint_head_config.json", "joint_head.safetensors"):
            shutil.copyfile(FIXTURE / name, self.path / name)

    def run_cli(self):
        return subprocess.run([str(BINARY), str(self.path), "--check"], text=True, capture_output=True, timeout=30)

    def rejected(self):
        result = self.run_cli()
        self.assertNotEqual(0, result.returncode)
        self.assertEqual("", result.stdout, "a failed check must not emit a success receipt")

    def test_complete_pair_has_verified_payload_hash_and_no_model_readiness_claim(self):
        self.pair()
        result = self.run_cli()
        self.assertEqual(0, result.returncode, result.stderr)
        record = json.loads(result.stdout)
        self.assertTrue(record["head_present"] and record["head_validated"])
        self.assertEqual(122, record["head_tensors"])
        self.assertEqual(16, record["backbone_hidden"])
        self.assertEqual(8, record["head_config"]["width"])
        self.assertEqual(hashlib.sha256((self.path / "joint_head.safetensors").read_bytes()).hexdigest(), record["head_file_sha256"])
        self.assertFalse(record["backbone_weights_checked"] or record["gpu"] or record["model_ready"])

    def test_plain_pack_has_no_validated_head_or_hash(self):
        result = self.run_cli()
        self.assertEqual(0, result.returncode, result.stderr)
        record = json.loads(result.stdout)
        self.assertFalse(record["head_present"] or record["head_validated"] or record["model_ready"])
        self.assertEqual(0, record["head_tensors"])
        self.assertIsNone(record["head_config"])
        self.assertIsNone(record["head_file_sha256"])

    def test_incomplete_pair_and_truncated_payload_are_errors(self):
        shutil.copyfile(FIXTURE / "joint_head_config.json", self.path / "joint_head_config.json")
        self.rejected()
        self.pair()
        payload = self.path / "joint_head.safetensors"
        payload.write_bytes(payload.read_bytes()[:-2])
        self.rejected()

    def test_incompatible_backbone_is_an_error(self):
        self.pair()
        (self.path / "config.json").write_text(json.dumps({**CONFIG, "hidden_size": 32}))
        self.rejected()


if __name__ == "__main__":
    unittest.main()
