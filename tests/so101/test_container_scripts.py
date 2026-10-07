"""Hermetic shell integration tests: no Docker daemon, downloads or GPU use."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def executable(path, body):
    path.write_text(body)
    path.chmod(0o755)


class ContainerTest(unittest.TestCase):
    def test_mounts_spaces_overrides_gpu_shared_memory_and_token_by_name(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            bin_dir = temp / "bin"
            bin_dir.mkdir()
            docker_log = temp / "docker.json"
            executable(
                bin_dir / "docker",
                f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ['FAKE_DOCKER_LOG']).write_text(json.dumps(sys.argv[1:]))\n",
            )
            env = dict(os.environ)
            env.update(
                PATH=str(bin_dir) + os.pathsep + env["PATH"],
                FAKE_DOCKER_LOG=str(docker_log),
                HF_HOME=str(temp / "HF cache"),
                UV_CACHE_DIR=str(temp / "uv cache"),
                HF_TOKEN="fake-secret-must-never-appear",
                OUTPUT_ROOT=str(temp / "wrong-output"),
            )
            out = temp / "chosen outputs"
            command = [
                str(ROOT / "run_container.sh"),
                "--image",
                "test:image",
                "--output_root",
                str(out),
                "--shm-size",
                "24g",
                "--",
                "/bin/bash",
                "-lc",
                "echo argument with spaces",
            ]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(docker_log.read_text())
            self.assertEqual(args[:6], ["run", "--rm", "--gpus", "all", "--shm-size", "24g"])
            self.assertIn(f"{ROOT}:/workspace/so101-groot-train", args)
            self.assertIn(f"{out}:/external_storage/models", args)
            self.assertIn(f"{temp / 'HF cache'}:/root/.cache/huggingface", args)
            self.assertIn(f"{temp / 'uv cache'}:/root/.cache/uv", args)
            self.assertIn("HF_TOKEN", args)
            self.assertIn("OUTPUT_ROOT=/external_storage/models/so101_official_n17", args)
            self.assertIn(f"SO101_HOST_OUTPUT_ROOT={out}", args)
            self.assertNotIn(env["HF_TOKEN"], " ".join(args) + result.stdout + result.stderr)
            self.assertEqual(
                args[-4:], ["test:image", "/bin/bash", "-lc", "echo argument with spaces"]
            )

    def test_help_and_bad_options_do_not_invoke_docker(self):
        result = subprocess.run(
            [str(ROOT / "run_container.sh"), "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("HF_HOME", result.stdout)
        for flags in (["--output_root"], ["--not-a-flag"]):
            result = subprocess.run(
                [str(ROOT / "run_container.sh"), *flags], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 2)


class SetupTest(unittest.TestCase):
    def fixture(self, temp, changed_torch=False):
        temp = Path(temp)
        repo = temp / "checkout with spaces"
        (repo / "so101/provenance").mkdir(parents=True)
        shutil.copyfile(ROOT / "setup_env.sh", repo / "setup_env.sh")
        (repo / "setup_env.sh").chmod(0o755)
        shutil.copyfile(
            ROOT / "so101/provenance/environment_versions.json",
            repo / "so101/provenance/environment_versions.json",
        )
        bin_dir = temp / "bin"
        bin_dir.mkdir()
        log = temp / "calls.jsonl"
        fake_python = f"""#!{sys.executable}
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ["FAKE_SETUP_LOG"]).open("a") as stream:
    stream.write(json.dumps(["python",*args])+"\\n")
if args==["-c", "import platform; print(platform.python_version())"]:
    print("3.12.13")
elif args==["-m","so101.setup_checks","--torch-fingerprint"]:
    count_path=Path(os.environ["FAKE_SETUP_COUNT"])
    count=int(count_path.read_text()) if count_path.exists() else 0
    count_path.write_text(str(count+1))
    value="changed" if {changed_torch!r} and count>0 else "same"
    print(json.dumps({{"version":"2.9.0+cu128","record":value}}))
elif args[:2]==["-m","so101.launcher"]:
    assert args[2:5]==["--stage","1","--dry_run"]
    assert os.environ["HF_HUB_OFFLINE"]=="1"
elif args==["-m","so101.weights","--download"]:
    assert os.environ.get("HF_HUB_OFFLINE")!="1"
elif args[:3]==["-m","so101.setup_checks","--cleanup-setup-checkpoints"]:
    assert Path(args[3]).name.startswith("so101-setup-dry-run-")
elif args!=["-m","so101.setup_checks"]:
    raise SystemExit("unexpected fake Python invocation")
"""
        fake_path = temp / "fake_python"
        executable(fake_path, fake_python)
        fake_uv = f"""#!{sys.executable}
import json,os,shutil,sys
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ["FAKE_SETUP_LOG"]).open("a") as stream:
    stream.write(json.dumps(["uv",*args])+"\\n")
if args[0]=="venv":
    assert "--python" in args and args[args.index("--python")+1]=="3.12.13"
    folder=Path.cwd()/".venv/bin"
    folder.mkdir(parents=True)
    shutil.copyfile(os.environ["FAKE_SETUP_PYTHON"],folder/"python")
    (folder/"python").chmod(0o755)
"""
        executable(bin_dir / "uv", fake_uv)
        env = dict(os.environ)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            env.pop(name, None)
        env.update(
            PATH=str(bin_dir) + os.pathsep + env["PATH"],
            FAKE_SETUP_LOG=str(log),
            FAKE_SETUP_PYTHON=str(fake_path),
            FAKE_SETUP_COUNT=str(temp / "fingerprints"),
            OUTPUT_ROOT=str(temp / "outputs"),
            HF_TOKEN="fake-secret-not-printed",
        )
        return repo, env, log

    def test_exact_pins_no_dependency_bnb_two_snapshots_and_official_dry_run(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, env, log = self.fixture(temp)
            result = subprocess.run(
                [str(repo / "setup_env.sh")], env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertIn(
                ["uv", "sync", "--frozen", "--no-dev", "--inexact", "--python", "3.12.13"],
                calls,
            )
            self.assertIn(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    ".venv/bin/python",
                    "--no-deps",
                    "bitsandbytes==0.50.2",
                    "av==16.0.1",
                ],
                calls,
            )
            self.assertIn(["python", "-m", "so101.weights", "--download"], calls)
            dry_calls = [call for call in calls if call[:3] == ["python", "-m", "so101.launcher"]]
            self.assertEqual(len(dry_calls), 1)
            self.assertTrue(Path(dry_calls[0][-1]).is_relative_to(Path(env["OUTPUT_ROOT"])))
            self.assertNotIn(env["HF_TOKEN"], result.stdout + result.stderr)
            self.assertIn("PASS setup complete", result.stdout)
            # Second setup preserves the already matching environment.
            result = subprocess.run(
                [str(repo / "setup_env.sh")], env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(sum(call[:2] == ["uv", "venv"] for call in calls), 1)

    def test_torch_replacement_aborts_before_weights_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, env, log = self.fixture(temp, changed_torch=True)
            result = subprocess.run(
                [str(repo / "setup_env.sh")], env=env, capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("torch changed", result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertFalse(
                any(
                    call[:3]
                    in (["python", "-m", "so101.weights"], ["python", "-m", "so101.launcher"])
                    for call in calls
                )
            )


if __name__ == "__main__":
    unittest.main()
