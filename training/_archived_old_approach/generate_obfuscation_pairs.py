#!/usr/bin/env python3
"""
Generate obfuscation pairs (obfuscated, clean) for JS deobfuscation training.
- Input: directory containing .js files (default: training/data/raw_js)
- Output: training/data/obfuscation_pairs.jsonl
- Uses terser if available (`npx terser`), otherwise falls back to a simple minifier.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw_js"
OUT_PATH = ROOT / "data" / "obfuscation_pairs.jsonl"


def terser_available() -> bool:
    try:
        subprocess.run(["npx", "terser", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def terser_obfuscate(code: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["npx", "terser", "--compress", "--mangle"],
            input=code.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return proc.stdout.decode()
    except Exception as e:
        print(f"[warn] terser failed: {e}")
        return None


def simple_minify(code: str) -> str:
    # Extremely light minify: strip comments/whitespace
    lines = []
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("//") or line == "":
            continue
        lines.append(line)
    return "".join(lines)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    js_files = list(RAW_DIR.glob("**/*.js"))
    if not js_files:
        print(f"❌ No JS files found in {RAW_DIR}. Add source JS files to proceed.")
        return 1

    use_terser = terser_available()
    print(f"🔧 terser available: {use_terser}")

    written = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for js_path in js_files:
            try:
                clean = js_path.read_text(encoding="utf-8")
            except Exception:
                continue

            obf = terser_obfuscate(clean) if use_terser else None
            if obf is None:
                obf = simple_minify(clean)

            record = {"obfuscated": obf, "clean": clean, "source": str(js_path.relative_to(ROOT))}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"✅ Generated {written} pairs -> {OUT_PATH}")
    print("下一步: 运行 train_seq2seq_deobfuscator.py 使用真实配对数据重新训练")
    return 0


if __name__ == "__main__":
    sys.exit(main())
