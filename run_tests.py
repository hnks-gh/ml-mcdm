import sys
import os
import subprocess

os.chdir(r"C:\Users\hoang\Documents\ml-mcdm")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-q"],
    capture_output=True,
    text=True,
    cwd=r"C:\Users\hoang\Documents\ml-mcdm"
)
output = result.stdout + result.stderr
with open(r"C:\Users\hoang\Documents\ml-mcdm\test_output.txt", "w") as f:
    f.write(output)
    f.write(f"\nExit code: {result.returncode}\n")
print("Done. Exit code:", result.returncode)
print(output[-3000:])
