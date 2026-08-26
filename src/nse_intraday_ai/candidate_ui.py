"""The intraday candidate's paper track, rendered for the app.

This screen exists to show a rule that is *not* being traded, which is an
unusual thing for a trading app to do and the reason it is a separate panel
rather than another row of tickets.  The candidate returned +67.44 bps over 14
in-sample sessions, passed a best-of-48 permutation null, a time-matched
control and a both-halves split, and then lost 23.51 bps over its first four
forward trades.  Both halves of that belong on the screen: a rule that looked
this good and still failed forward is the clearest available illustration of
why the go-live gate is worth obeying.

Nothing here places or sizes an order.  The rupee column is what the trade
*would* have made at the standard position, so the log can be scored against
`intraday_go_live()` as sessions accumulate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nse_intraday_ai.execution_plan import GO_LIVE_GATE, FORWARD_TEST, intraday_go_live

LOG = Path(__file__).resolve().parents[2] / "data" / "candidate_paper.csv"

RULE = """
**Ridge, top 1% of its own trailing distribution, one trade a day.**

| | |
|---|---|
| Model | `Ridge(alpha=10.0)`, refit each session on every labelled session before it |
| Universe | top-150 liquid NSE names |
| Threshold | 99th percentile of the model's scores on its own training data |
| Entry | the **first** signal of the day above that threshold — not the best of the day |
| Cap | one trade, then stop for the session |
| Structure | stop 1.5x ATR, target 3.0x ATR, exit after 60 min or 15:15 |

The threshold is *negative* — ridge scores nearly everything below zero, so
"top 1%" means the least-bad 1%, not a confident profit forecast. Entry is
first-to-print rather than best-of-day because ranking a whole session uses
signals that had not printed yet, which is the look-ahead that produced the
retracted +27.62 bps result.
"""


def _gate_from_log(d: pd.DataFrame) -> tuple[bool, dict[str, bool], np.ndarray]:
    traded = d[d["traded"] == 1].dropna(subset=["net_bps"])
    v = traded["net_bps"].to_numpy(float)
    if v.size == 0:
        return False, {}, v
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(v, v.size, replace=True).mean() for _ in range(5000)])
    half = v.size // 2
    ok, checks = intraday_go_live(
        net_bps=float(v.mean()),
        sessions=len(d),
        p_value=float(np.mean(boot <= 0)),
        half1_bps=float(v[:half].mean()) if half else 0.0,
        half2_bps=float(v[half:].mean()),
        cells_scanned=1,
    )
    return ok, checks, v


def render() -> None:
    st.header("Intraday candidate — paper track")
    st.error(
        "**Not tradeable.** This rule is being observed, not traded. It lost "
        f"{abs(FORWARD_TEST['forward_bps']):.1f} bps over its first "
        f"{FORWARD_TEST['forward_trades']} forward trades after returning "
        f"+{FORWARD_TEST['in_sample_bps']:.1f} bps over "
        f"{FORWARD_TEST['in_sample_sessions']} in-sample sessions.",
        icon="🚫",
    )
    st.markdown(RULE)

    if not LOG.exists():
        st.info(
            "No paper log yet. Run `python scripts/candidate_paper.py "
            "--backfill 2026-08-18 2026-08-25` to seed it, or wait for the "
            "`nse-candidate-paper` timer to record tonight's session."
        )
        return

    d = pd.read_csv(LOG)
    traded = d[d["traded"] == 1].dropna(subset=["net_bps"])
    ok, checks, v = _gate_from_log(d)

    cols = st.columns(4)
    cols[0].metric("Sessions observed", len(d))
    cols[1].metric("Trades taken", len(traded))
    if v.size:
        cols[2].metric("Mean net", f"{v.mean():+.1f} bps")
        cols[3].metric("Total", f"Rs {traded['rupees'].sum():+,.0f}")

    st.caption(
        f"{len(d) - len(traded)} session(s) produced no trade because nothing "
        "cleared the threshold. That is part of the rule, not a gap in the data."
    )

    if not traded.empty:
        st.dataframe(
            traded[["day", "symbol", "side", "score", "threshold", "net_bps", "rupees"]],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Go-live gate")
    if ok:
        st.success("All checks pass. This is now worth a funded discussion.", icon="✅")
    else:
        st.warning("Not cleared — the rule stays on paper.", icon="⏳")
    for name, passed in checks.items():
        st.write(("✅ " if passed else "❌ ") + name)

    remaining = max(0, GO_LIVE_GATE["min_sessions"] - len(d))
    if remaining:
        st.info(
            f"**{remaining} more sessions** before the sample is large enough for "
            "the question to be answerable either way. The `nse-candidate-paper` "
            "timer records one per weekday at 16:10.",
            icon="📅",
        )
