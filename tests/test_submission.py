import numpy as np
import pandas as pd

from desfecho.submission import build_submission, read_submission, validate_submission, write_submission


def test_desfecho_usa_a_probabilidade_gravada(tmp_path):
    ids = ["A1", "A2", "A3"]
    probs = np.array([0.1999994, 0.1999996, 0.91])
    frame = build_submission(ids, probs, threshold=0.2)
    assert frame["desfecho"].tolist() == [0, 1, 1]
    path = tmp_path / "sub.csv"
    write_submission(frame, path)
    loaded = read_submission(path)
    validate_submission(loaded, 0.2, expected_ids=ids)
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "id,prob,desfecho"
    assert text[1].startswith("A1,0.199999,0")


def test_rejeita_limiar_inconsistente():
    frame = pd.DataFrame({"id": ["A1"], "prob": [0.4], "desfecho": [1]})
    try:
        validate_submission(frame, 0.5)
    except ValueError as exc:
        assert "limiar" in str(exc)
    else:
        raise AssertionError("deveria ter recusado")
