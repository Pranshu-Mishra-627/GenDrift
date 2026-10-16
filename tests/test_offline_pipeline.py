from pathlib import Path
import tempfile
from core.offline_pipeline import run_offline_experiment


def main():
    root = Path(__file__).resolve().parents[1]
    iris = root / "data" / "iris.csv"
    with tempfile.TemporaryDirectory() as tmp:
        output = run_offline_experiment(
            iris, "target", generations=1, output_dir=tmp, track_shap=False
        )
        assert Path(output["metrics_csv"]).exists()
        assert Path(output["metrics_json"]).exists()
        assert len(output["results"]) == 1
        assert output["results"][0]["generation"] == 0
    print("Offline pipeline tests passed.")


if __name__ == "__main__":
    main()
