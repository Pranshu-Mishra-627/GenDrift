"""
Run every test_*.py file in the GenDrift project and save all output to results.txt.

Place this file at the project root, alongside test_generation.py and the tests/ folder.
Run with: python run_all_tests.py
"""
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / "results.txt"


def discover_tests():
    """Find root-level tests and tests inside the tests/ directory."""
    found = []

    # Root-level scripts such as test_generation.py and test_trainer.py
    found.extend(
        path for path in ROOT.glob("test_*.py")
        if path.resolve() != Path(__file__).resolve()
    )

    # Test modules under tests/, including future nested test folders
    tests_dir = ROOT / "tests"
    if tests_dir.exists():
        found.extend(tests_dir.rglob("test_*.py"))

    # Stable order makes the report easier to compare between runs.
    return sorted(set(found), key=lambda path: str(path.relative_to(ROOT)).lower())


def command_for(test_path):
    relative = test_path.relative_to(ROOT)

    # Use module execution for files directly inside tests/ so imports resolve
    # consistently from the project root.
    if relative.parts[0] == "tests" and len(relative.parts) == 2:
        module_name = f"tests.{test_path.stem}"
        return [sys.executable, "-m", module_name]

    # Root-level scripts and nested test scripts are executed by file path.
    return [sys.executable, str(test_path)]


def main():
    tests = discover_tests()
    started = datetime.now()

    lines = [
        "GENDRIFT TEST RUN REPORT",
        f"Started: {started.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python: {sys.version.replace(chr(10), ' ')}",
        f"Project: {ROOT}",
        f"Discovered tests: {len(tests)}",
        "=" * 80,
        "",
    ]

    if not tests:
        lines.append("ERROR: No test_*.py files were found.")
        OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"No tests found. Details saved to {OUTPUT_FILE}")
        return 2

    passed = 0
    failed = 0

    for index, test_path in enumerate(tests, start=1):
        relative = test_path.relative_to(ROOT)
        command = command_for(test_path)
        display_command = " ".join(
            f'"{part}"' if " " in part else part for part in command
        )

        print(f"[{index}/{len(tests)}] Running {relative} ...", flush=True)
        lines.extend([
            "",
            "=" * 80,
            f"TEST {index}/{len(tests)}: {relative}",
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Command: {display_command}",
            "-" * 80,
        ])

        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace",
            )
            lines.append(f"Exit code: {result.returncode}")
            lines.append("STATUS: " + ("PASS" if result.returncode == 0 else "FAIL"))

            if result.stdout:
                lines.extend(["", "[STDOUT]", result.stdout.rstrip()])
            if result.stderr:
                lines.extend(["", "[STDERR]", result.stderr.rstrip()])

            if result.returncode == 0:
                passed += 1
                print("    PASS", flush=True)
            else:
                failed += 1
                print(f"    FAIL (exit code {result.returncode})", flush=True)

        except Exception:
            failed += 1
            lines.extend([
                "STATUS: FAIL",
                "",
                "[RUNNER ERROR]",
                traceback.format_exc(),
            ])
            print("    FAIL (runner error)", flush=True)

        # Keep the report updated after each test in case a later test crashes.
        summary = [
            "",
            "=" * 80,
            "SUMMARY SO FAR",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Total:  {len(tests)}",
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        OUTPUT_FILE.write_text("\n".join(lines + summary) + "\n", encoding="utf-8")

    finished = datetime.now()
    lines.extend([
        "",
        "=" * 80,
        "FINAL SUMMARY",
        f"Started:  {started.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Finished: {finished.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Passed:   {passed}",
        f"Failed:   {failed}",
        f"Total:    {len(tests)}",
        "Overall:  " + ("PASS" if failed == 0 else "FAIL"),
    ])
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Finished: {passed} passed, {failed} failed, {len(tests)} total.")
    print(f"Full output saved to: {OUTPUT_FILE}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
