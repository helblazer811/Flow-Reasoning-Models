import pytest
from frm import evaluate
import evaluate_adaptive


def test_default_dispatches_calibrated_restarts(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(evaluate_adaptive, "main", lambda argv: calls.append(argv))
    evaluate.main(
        [
            "--checkpoint",
            "model.pt",
            "--benchmark",
            "test.npz",
            "--calibration",
            "threshold.json",
            "--resume",
        ]
    )
    assert calls and "threshold.json" in calls[0] and "--resume" in calls[0]


def test_default_requires_calibration(capsys):
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--checkpoint", "model.pt"])
    assert error.value.code == 2
    assert "requires --benchmark and --calibration" in capsys.readouterr().err


def test_fixed_depth_requires_config():
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--checkpoint", "model.pt", "--sampler", "fixed_depth"])
    assert error.value.code == 2


def test_local_bundle_checksums_and_evaluation(monkeypatch, tmp_path):
    import json
    from frm.evaluate import evaluate
    from frm.adaptive import sha256

    task = tmp_path / "sudoku"
    task.mkdir()
    model = task / "model.pt"
    blob = tmp_path / "blob"
    blob.write_bytes(b"checkpoint")
    model.symlink_to(blob)
    (task / "manifest.json").write_text(
        json.dumps(dict(training_seed=0, files={"model.pt": sha256(model)}))
    )

    def fake_run(argv):
        output = __import__("pathlib").Path(argv[argv.index("--output-dir") + 1])
        output.mkdir()
        (output / "evaluation.json").write_text(json.dumps({"policies": []}))

    monkeypatch.setattr(evaluate_adaptive, "main", fake_run)
    result = evaluate(task, output_dir=tmp_path / "result", max_samples=8)
    assert result["policies"] == []
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate(task, output_dir=tmp_path / "result2")


def test_relative_calibration_uses_snapshot_directory(tmp_path):
    blob = tmp_path / "blob"
    blob.write_text("{}")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    threshold = snapshot / "threshold.json"
    threshold.symlink_to(blob)
    expected = snapshot / "calibration.npz"
    assert (
        evaluate_adaptive.calibration_benchmark_path(threshold, "calibration.npz")
        == expected
    )
    assert (
        evaluate_adaptive.calibration_benchmark_path(threshold, str(expected))
        == expected
    )


def test_bundle_cli_dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        evaluate, "evaluate", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    evaluate.main(
        ["--model-dir", str(tmp_path), "--max-samples", "8", "--device", "cpu"]
    )
    assert calls[0][0] == (tmp_path,)
    assert calls[0][1]["max_samples"] == 8
    assert calls[0][1]["device"] == "cpu"


def test_bundle_cli_rejects_fixed_depth():
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--model-dir", "models/sudoku", "--sampler", "fixed_depth"])
    assert error.value.code == 2
