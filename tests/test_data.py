import hashlib

import numpy as np
import pytest

from rakshpulse.data import dataset_loader
from rakshpulse.data.download import DATASETS, file_checksum


def test_registry_has_licence_checksum_and_citation():
    for spec in DATASETS.values():
        assert spec.license and spec.citation and spec.doi
        for f in spec.files:
            assert f.algo in ("sha256", "md5") and len(f.checksum) in (32, 64) and f.size > 0


def test_file_checksum(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"rakshpulse")
    assert file_checksum(p, "sha256") == hashlib.sha256(b"rakshpulse").hexdigest()
    assert file_checksum(p, "md5") == hashlib.md5(b"rakshpulse").hexdigest()


@pytest.fixture
def guilin_fixture(tmp_path):
    root = tmp_path / "guilin"
    (root / "Raw_data").mkdir(parents=True)
    (root / "information.csv").write_text(
        "ID,Age,Gender,Height/cm,Weight/kg,SBP/mmHg,DBP/mmHg,HR/bpm\n001,24,Male,176,72,128,80,70\n"
        "002,22,Female,160,50,105,65,75\n",
        encoding="utf-8",
    )
    rng = np.random.default_rng(0)
    for sid in ("001", "002"):
        for st in ("1", "2"):
            rows = ["ECG_I,PPG_RED,PPG_IR,{'Hz': 250}"]
            for i in range(500):
                rows.append(f"{280000 + i},{400000 + int(rng.integers(100))},{430000 + i % 50},")
            (root / "Raw_data" / f"{sid}_{st}.csv").write_text("\n".join(rows) + "\n")
    return root


def test_guilin_info_and_raw_reader(guilin_fixture):
    info = dataset_loader.load_guilin_info(guilin_fixture)
    assert info["001"]["sbp"] == 128 and info["002"]["sex"] == 0.0
    ch = dataset_loader.read_guilin_raw(guilin_fixture / "Raw_data" / "001_1.csv")
    assert set(ch) == {"ecg", "red", "ir"} and ch["ir"].size == 500


def test_guilin_only_rest_state_is_labelled(guilin_fixture):
    recs = list(dataset_loader.iter_guilin(states=("1", "2"), root=guilin_fixture))
    assert len(recs) == 4
    for r in recs:
        if r.state == "rest":
            assert r.sbp is not None
        else:
            assert r.sbp is None and r.dbp is None
        assert r.fs == 250.0 and r.has_dc
