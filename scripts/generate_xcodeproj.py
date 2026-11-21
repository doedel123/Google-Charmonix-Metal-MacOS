#!/usr/bin/env python3
"""
Deterministic Xcode project generator for ChamonixSkiSim.
Uses `xcodegen` against ChamonixSkiSim/project.yml, which automatically discovers
all `.swift` and `.metal` files under Sources/, Shaders/, and Tests/, and bundles
`Resources/assets/` into `Contents/Resources/assets/`.
"""
import os
import subprocess
import sys
from pathlib import Path

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    spec_path = project_root / "project.yml"

    # Ensure source directories exist so xcodegen never warns on missing folders
    for rel in [
        "Sources/Core",
        "Sources/Terrain",
        "Sources/Simulation",
        "Sources/Rig",
        "Sources/World",
        "Sources/PostFX",
        "Sources/Audio",
        "Sources/App",
        "Shaders",
        "Tests",
    ]:
        (project_root / rel).mkdir(parents=True, exist_ok=True)

    if not spec_path.exists():
        print(f"Error: {spec_path} not found", file=sys.stderr)
        return 1

    cmd = ["xcodegen", "generate", "--spec", str(spec_path), "--project", str(project_root)]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(project_root))
    return res.returncode

if __name__ == "__main__":
    sys.exit(main())
