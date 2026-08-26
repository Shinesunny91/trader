"""Findings this workspace published and later withdrew.

A backtest number, once written into a comment or a README, outlives the run
that produced it.  It gets quoted by the next script, defended by the next
change, and eventually nobody remembers whether anyone re-tested it.  The cure
is to keep the withdrawal next to the claim and make the code say so out loud
every time the artefact is loaded.

Nothing here changes behaviour.  It exists so that a retracted result cannot be
quietly re-quoted as a live one.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Retraction:
    claim: str
    verdict: str
    detail: str

    def banner(self) -> str:
        rule = "!" * 78
        return (f"\n{rule}\nRETRACTED CLAIM — {self.claim}\n  {self.verdict}\n"
                f"{self.detail}\n{rule}\n")


MODEL_PREDICTIONS = Retraction(
    claim="top-1 model-ranked signal returned +27.62 bps over 34 sessions",
    verdict="WITHDRAWN 2026-08-19. Do not quote this file's headline number.",
    detail=(
        "  * data/model_predictions.parquet holds the argmax of four model families,\n"
        "    selected on the same held-out sessions used to score them.\n"
        "  * Re-run over 48 sessions, every family loses money on its top pick:\n"
        "      ridge -6.91   hgb_shallow -8.19   hgb_deep -22.14   rf -20.53 bps\n"
        "    rf is the family that was shipped; +27.62 was rf on a 34-session subset.\n"
        "  * The population has no edge to rank: gross +0.81 bps against a 10.10 bps\n"
        "    round trip, and 0 of 41 features have a best decile that clears cost\n"
        "    out of sample.  See scripts/intraday_edge_audit.py.\n"
        "  Anything derived from this file — exit sweeps, cap comparisons — inherits\n"
        "  the defect, because it was tuned over a book that was ranking noise."
    ),
)
