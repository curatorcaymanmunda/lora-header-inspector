"""Exercise an installed distribution without importing the source checkout.

Invoke with the target virtualenv's Python and -I, from outside the checkout.
Only synthetic files in a temporary directory are inspected; no apply/undo
commands or real LoRA libraries are used.
"""
import json
import struct
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import lora_header_inspector


def run(*args):
    return subprocess.check_output(args, text=True)


def main():
    installed = Path(lora_header_inspector.__file__).resolve()
    site = Path(sysconfig.get_path("purelib")).resolve()
    if site not in installed.parents:
        raise RuntimeError(f"import escaped installed site-packages: {installed}")
    scripts = Path(sysconfig.get_path("scripts"))
    for name in ("lora-inspect", "lora-sort", "lora-smoke", "lora-fixtures"):
        run(str(scripts / name), "--help")
    run(sys.executable, "-I", "-m", "lora_header_inspector", "--version")
    report = json.loads(run(str(scripts / "lora-fixtures"), sys.argv[1], "--json"))
    if not report["fixture_count"] or report["failed"]:
        raise RuntimeError(f"fixture validation failed: {report}")
    with tempfile.TemporaryDirectory() as temp:
        model = Path(temp) / "synthetic.safetensors"
        header = json.dumps({"__metadata__": {}, "synthetic.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
        }}).encode("utf-8")
        original = struct.pack("<Q", len(header)) + header + b"\0" * 4
        model.write_bytes(original)
        records = json.loads(run(str(scripts / "lora-smoke"), str(model), "--json"))
        if len(records) != 1 or records[0].get("status") == "error":
            raise RuntimeError(f"installed smoke inspection failed: {records}")
        if model.read_bytes() != original or list(Path(temp).iterdir()) != [model]:
            raise RuntimeError("read-only smoke inspection changed the temporary library")
    print(f"Installed package verified: {installed}; {report['passed']} fixtures passed")


if __name__ == "__main__":
    main()
