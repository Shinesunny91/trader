#!/usr/bin/env bash
# Weekly: refresh the macro panel, rebuild the labelled dataset, and retrain the
# ranking model. train_signal_model.py refuses to write unless the model beats
# the incumbent on walk-forward, so a failed week simply leaves the old model in
# place rather than shipping a worse one.
#
# Retraining matters here specifically because the measured risk is *decay*: the
# earlier model run made most of its money in its first three weeks.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== $(date -Is) weekly retrain ==="
.venv/bin/python scripts/backfill_context.py --period 60d || echo "context backfill failed"
.venv/bin/python scripts/build_dataset.py --workers 20 --since "$(date -d '90 days ago' +%Y-%m-%d)" \
    || { echo "dataset build failed; keeping the existing model"; exit 1; }
.venv/bin/python scripts/train_signal_model.py
echo "=== $(date -Is) done ==="
