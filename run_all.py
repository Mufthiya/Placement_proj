import argparse
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
PORT = 5050

REQUIRED_ARTIFACTS = [
    "physical_health_summary.json",
    "mental_health_summary.json",
    "physical_health_linear.pkl",
    "physical_health_rf.pkl",
    "physical_health_xgboost.json",
    "mental_health_linear.pkl",
    "mental_health_rf.pkl",
    "mental_health_xgboost.json",
]


def artifacts_exist():
    return all((OUTPUT_DIR / name).exists() for name in REQUIRED_ARTIFACTS)


def train_models(with_navigation=False):
    print("=" * 60)
    print("STEP 1: training models (50 epochs each)" if not with_navigation
          else "STEP 1: training models + MNLI demo")
    print("=" * 60)
    cmd = (
        "from main import run_pipeline; "
        f"run_pipeline(run_navigation={with_navigation})"
    )
    result = subprocess.run([sys.executable, "-c", cmd], cwd=str(ROOT))
    if result.returncode != 0:
        sys.exit("Training failed. Fix the error above and re-run run_all.py")


def run_cli():
    print("Starting interactive command-line assessment...")
    subprocess.run([sys.executable, str(ROOT / "realtime_app.py")], cwd=str(ROOT))


def open_browser(url):
    time.sleep(4)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def launch_ui(port, no_browser=False):
    url = f"http://127.0.0.1:{port}"
    print()
    print("=" * 60)
    print(f"LIFE UI starting at {url}")
    print("=" * 60)
    if not no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    import ui_app
    ui_app.app.run(host="127.0.0.1", port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="Life - run everything")
    parser.add_argument("--train", action="store_true",
                        help="force retraining even if artifacts exist")
    parser.add_argument("--cli", action="store_true",
                        help="use the terminal assessment instead of the web UI")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.cli:
        if not artifacts_exist() or args.train:
            train_models()
        run_cli()
        return

    if args.train or not artifacts_exist():
        if not artifacts_exist():
            print("Model artifacts missing - training first...")
        train_models()
    else:
        print("Model artifacts found - skipping training (use --train to retrain).")

    launch_ui(args.port, args.no_browser)


if __name__ == "__main__":
    main()
