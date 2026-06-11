"""Add cot_shorts column to data/per_question/nonthinking.csv.

Reads the CoT-SC sweep output (data/raw/exp_cot_sc/{dataset}/{model}.json)
and merges per-question extracted short answers as a new `cot_shorts`
column, JSON-encoded list of 10 strings (parallel to existing nv_answers).

Existing columns (model, dataset, question_id, op_direct, op_expansion,
nv_answers, ground_truth, aliases) are preserved verbatim.

After this, analyses can compute pairwise K=3, K=10, semantic entropy, MV
correctness etc. on cot_shorts via the same helpers used for nv_answers.

Usage:
    python scripts/merge_cot_sc_into_csv.py
"""
from __future__ import annotations

import csv, json, os, sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH  = REPO_ROOT / 'data' / 'per_question' / 'nonthinking.csv'
COT_ROOT  = REPO_ROOT / 'data' / 'raw' / 'exp_cot_sc'
TMP_PATH  = REPO_ROOT / 'data' / 'per_question' / 'nonthinking.csv.tmp'


def load_cot_shorts():
    """Returns {(model, dataset, qid): [10 short answers]}."""
    if not COT_ROOT.is_dir():
        sys.exit(f"ERROR: CoT-SC data not found at {COT_ROOT}")
    out = {}
    for ds_dir in sorted(COT_ROOT.iterdir()):
        if not ds_dir.is_dir(): continue
        ds = ds_dir.name
        for f in sorted(ds_dir.iterdir()):
            if not f.name.endswith('.json'): continue
            model = f.stem
            data = json.loads(f.read_text())
            for r in data['results']:
                out[(model, ds, r['question_id'])] = r['cot_shorts']
    return out


def main():
    print(f"Loading CoT-SC shorts from {COT_ROOT}...")
    cot = load_cot_shorts()
    print(f"  loaded {len(cot)} (model, dataset, qid) cells")

    print(f"Reading existing CSV: {CSV_PATH}")
    with open(CSV_PATH) as fin:
        rdr = csv.DictReader(fin)
        fieldnames = list(rdr.fieldnames)
        if 'cot_shorts' in fieldnames:
            sys.exit("ERROR: 'cot_shorts' column already present; nothing to do.")
        new_fieldnames = fieldnames[:fieldnames.index('ground_truth')] + \
                         ['cot_shorts'] + \
                         fieldnames[fieldnames.index('ground_truth'):]
        rows = list(rdr)
    print(f"  {len(rows)} rows, schema: {fieldnames}")

    # Join
    matched = unmatched = 0
    by_cell_count = defaultdict(int)
    for r in rows:
        key = (r['model'], r['dataset'], r['question_id'])
        shorts = cot.get(key)
        if shorts is None:
            unmatched += 1
            by_cell_count[(r['model'], r['dataset'])] += 1
            r['cot_shorts'] = json.dumps([])
        else:
            matched += 1
            r['cot_shorts'] = json.dumps(shorts)

    print(f"  matched {matched} rows, unmatched {unmatched}")
    if unmatched:
        print("  unmatched per (model, dataset):")
        for k, v in sorted(by_cell_count.items()):
            print(f"    {k}: {v}")

    print(f"Writing to tmp: {TMP_PATH}")
    with open(TMP_PATH, 'w', newline='') as fout:
        w = csv.DictWriter(fout, fieldnames=new_fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Atomic replace
    os.replace(TMP_PATH, CSV_PATH)
    print(f"  wrote {len(rows)} rows with new schema: {new_fieldnames}")
    print("Done.")


if __name__ == '__main__':
    main()
