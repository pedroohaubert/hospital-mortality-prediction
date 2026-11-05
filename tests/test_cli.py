from pathlib import Path

from typer.testing import CliRunner

from desfecho.cli import app
from tests.support import synthetic_frame

runner = CliRunner()


def test_ajuda_lista_os_comandos():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("train", "evaluate", "predict", "visualize"):
        assert name in result.stdout


def test_train_evaluate_predict(tmp_path: Path):
    frame = synthetic_frame(160, seed=9)
    data = tmp_path / "treino.csv"
    frame.to_csv(data, index=False)
    out = tmp_path / "artifacts"
    trained = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(data),
            "--out",
            str(out),
            "--splits",
            "2",
            "--repeats",
            "2",
            "--iterations",
            "20",
            "--threshold-boot",
            "12",
            "--auc-boot",
            "15",
        ],
    )
    assert trained.exit_code == 0, trained.stdout + trained.stderr
    evaluated = runner.invoke(app, ["evaluate", "--artifacts", str(out)])
    assert evaluated.exit_code == 0, evaluated.stdout
    assert "Limiar" in evaluated.stdout
    features = frame.drop(columns=["DESFECHO"])
    test = tmp_path / "teste.csv"
    features.to_csv(test, index=False)
    submission = tmp_path / "submissao.csv"
    predicted = runner.invoke(
        app,
        ["predict", "--data", str(test), "--model", str(out / "modelo.joblib"), "--out", str(submission)],
    )
    assert predicted.exit_code == 0, predicted.stdout + predicted.stderr
    text = submission.read_text(encoding="utf-8").splitlines()
    assert text[0] == "id,prob,desfecho"
    assert len(text) == len(frame) + 1
    drawn = runner.invoke(app, ["visualize", "--artifacts", str(out), "--out", str(tmp_path / "figs"), "--data", str(data)])
    assert drawn.exit_code == 0, drawn.stdout + drawn.stderr
    assert (tmp_path / "figs" / "roc.png").exists()
