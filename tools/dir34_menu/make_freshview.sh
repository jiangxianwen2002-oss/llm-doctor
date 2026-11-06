#!/bin/bash
# analyze86.py imports t1_run from a "fresh view" of t1_premise whose data/ folder is the fresh-300 evaluation set.
# Run from the repository root before analyze86.py.
set -e
mkdir -p t1_premise_freshview
cp t1_premise/t1_run.py t1_premise/t1_run_s2.py t1_premise/ddx_signal.py t1_premise/t1_run_sig.py t1_premise_freshview/
ln -sfn ../t1_premise/data_fresh t1_premise_freshview/data
echo "ok: t1_premise_freshview (data -> t1_premise/data_fresh)"
