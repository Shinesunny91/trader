"""Per-universe scan config: isolation, migration, defaults."""
from __future__ import annotations

import json

import pytest

import nse_intraday_ai.scan_config as sc


def test_universes_have_distinct_validated_gates():
    nse, com = sc.DEFAULTS_BY_UNIVERSE["nse"], sc.DEFAULTS_BY_UNIVERSE["commodity"]
    assert (nse["min_confidence"], nse["min_agreeing_votes"], nse["min_vote_share"]) == (70.0, 2, 0.50)
    # 70.0 since 2026-08-17: the 85 gate was calibrated on data pooled with an
    # April backfill and does not hold on live-only data — it produced 29
    # signals in seven weeks and inverted the edge on the three contracts that
    # carry it (RB=F/HO=F/SB=F: +43.49 bps ungated, -1.68 at conf>=85).
    assert (com["min_confidence"], com["min_agreeing_votes"], com["min_vote_share"]) == (70.0, 1, 0.70)
    assert nse["estimated_cost_bps"] == 15.0 and com["estimated_cost_bps"] == 5.0
    # 5m needs multi-session history for the 35-70 bar strategy guards
    for cfg in (nse, com):
        assert (cfg["interval"], cfg["period"]) == ("5m", "5d")


def test_saving_commodity_does_not_clobber_nse(tmp_path):
    """The 2026-07-07 incident: a commodity session must not poison NSE gates."""
    path = tmp_path / "scan_config.json"
    nse_cfg = sc.load("nse", path)
    sc.save({**sc.load("commodity", path), "min_confidence": 92.0}, "commodity", path)
    assert sc.load("nse", path) == nse_cfg
    assert sc.load("commodity", path)["min_confidence"] == 92.0


def test_legacy_flat_file_seeds_only_nse(tmp_path):
    path = tmp_path / "scan_config.json"
    path.write_text(json.dumps({"min_confidence": 61.0, "interval": "1m"}))
    assert sc.load("nse", path)["min_confidence"] == 61.0
    # commodity must NOT inherit foreign flat gates
    assert sc.load("commodity", path)["min_confidence"] == 70.0
    assert sc.load("commodity", path)["interval"] == "5m"   # not the flat file's 1m


def test_save_migrates_legacy_flat_file(tmp_path):
    path = tmp_path / "scan_config.json"
    path.write_text(json.dumps({"min_confidence": 61.0}))
    sc.save({"min_confidence": 90.0}, "commodity", path)
    raw = json.loads(path.read_text())
    assert raw["nse"]["min_confidence"] == 61.0
    assert raw["commodity"]["min_confidence"] == 90.0


def test_unknown_universe_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown universe"):
        sc.load("crypto", tmp_path / "x.json")
    with pytest.raises(ValueError, match="unknown universe"):
        sc.save({}, "crypto", tmp_path / "x.json")


def test_round_trip_through_risk_and_ensemble_configs(tmp_path):
    cfg = sc.load("nse", tmp_path / "scan_config.json")
    risk = sc.to_risk_config(cfg)
    ens = sc.to_ensemble_config(cfg)
    assert risk.min_confidence == 70.0 and risk.estimated_cost_bps == 15.0
    assert ens.min_agreeing_votes == 2 and ens.min_vote_share == 0.50
