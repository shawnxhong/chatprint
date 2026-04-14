"""
Sets up MSVC environment variables by running vcvarsall.bat, then
launches the actual test in a subprocess with those env vars.
Run this instead of test.py directly.
"""
import os
import subprocess
import sys

VCVARSALL = (
    r"C:\Program Files\Microsoft Visual Studio\2022\Community"
    r"\VC\Auxiliary\Build\vcvarsall.bat"
)

def get_msvc_env():
    """Run vcvarsall.bat x64 and capture the resulting environment."""
    cmd = f'"{VCVARSALL}" x64 > nul 2>&1 && set'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            env[key] = val
    return env

print("Setting up MSVC x64 environment via vcvarsall.bat...")
msvc_env = get_msvc_env()

if not msvc_env:
    print("ERROR: vcvarsall.bat produced no output. Check VS 2022 installation.")
    sys.exit(1)

# Verify cl.exe is now discoverable
import shutil
cl = shutil.which("cl", path=msvc_env.get("PATH", ""))
print(f"cl.exe found at: {cl}")

# Merge MSVC env into current env, then run the actual test
env = {**os.environ, **msvc_env}
env["HF_DATASETS_OFFLINE"] = "1"

test_script = os.path.join(os.path.dirname(__file__), "test.py")
print(f"Launching: {sys.executable} {test_script}\n")

proc = subprocess.run([sys.executable, test_script], env=env)
sys.exit(proc.returncode)
