"""Figure 3 (fig:empirics): one row per uncertainty-baseline family.

Updated for the arXiv version to include canonical CoT-SC (Wang et al. 2023).
Layout: 4 rows x 4 datasets:
  Row 1: Operadic consistency (OC) -- the proposal
  Row 2: CoT-SC K=10              -- dominant sample-aggregate baseline
  Row 3: -H K=10 (semantic entropy) -- cluster-based confidence
  Row 4: P(True)                   -- self-evaluation

Each subplot scatters per-(model, dataset) accuracy (x) vs the row's
signal-rate (y), across the twelve non-thinking models. Trend line and
Pearson r with two-sided p shown per subplot.

Naive (direct-answer) SC and decomposed-SC live in App. B.14's cross-model
table rather than this figure; including all baselines would push the
figure to 8+ rows.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy import stats

from analysis.scoring import (score_official, is_correct_official,
                              pairwise_official, cluster_entropy)
from analysis.data_loader import (load_nonthinking, load_baseline,
                                  NT_MODELS, NT_DATASETS)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / 'results'

MODEL_LABELS = {
    'cogito_671b': 'Cogito 671B', 'deepseek_v3_1': 'DeepSeek V3.1',
    'gemma3n_4b': 'Gemma3n 4B', 'liquidai_24b': 'LiquidAI 24B',
    'llama33_70b': 'Llama3.3 70B', 'llama8b': 'Llama3 8B',
    'qwen3_235b': 'Qwen3 235B', 'qwen7b': 'Qwen2.5 7B',
    'gpt_4o_mini': 'GPT-4o-mini', 'gemini_2_5_flash': 'Gemini 2.5 Flash',
    'llama_4_maverick': 'Llama-4 Maverick', 'mistral_large_2': 'Mistral Large 2',
}
DS_LABELS = {
    'hotpotqa': 'HotpotQA', 'musique': 'MuSiQue',
    'strategyqa': 'StrategyQA', 'drop': 'DROP',
}

COLORS = plt.cm.tab20.colors
OC_COLOR = '#1f4e79'
BASELINE_COLOR = '#7f7f7f'
# Panel tint for cells whose correlation is not significant at alpha = 0.05.
NONSIG_FACE = '#fcecec'

ROWS = [
    ('op_cons',   'Operadic consistency',               OC_COLOR),
    ('cot_k10',   'CoT-SC ($K{=}10$)',                  BASELINE_COLOR),
    ('neg_H_k10', '$-H$ ($K{=}10$, semantic entropy)',  BASELINE_COLOR),
    ('p_true',    '$P(\\text{True})$',                  BASELINE_COLOR),
]


def _f(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def per_cell():
    """Per (model, dataset) cell, compute accuracy plus row-key signal rates."""
    p_true_b = load_baseline('p_true')
    raw = defaultdict(lambda: {'n': 0, 'oc_count': 0, 'correct': 0,
                               'cot_sum': 0.0, 'cot_n': 0,
                               'H_sum': 0.0, 'H_n': 0,
                               'pt_sum': 0.0, 'pt_n': 0})
    for r in load_nonthinking():
        ds = r['dataset']; stem = r['model']; qid = r['question_id']
        c = raw[(stem, ds)]
        direct = r['op_direct']; exp = r['op_expansion']
        nv = r['nv_answers']; cot = r['cot_shorts']
        op_f1 = score_official(direct, exp, ds) if direct.strip() and exp.strip() else 0
        c['n'] += 1
        if op_f1 >= 0.5:
            c['oc_count'] += 1
        if is_correct_official(direct, r['ground_truth'], ds, r['aliases']):
            c['correct'] += 1

        cot_ne = [s for s in cot if s.strip()]
        if len(cot_ne) >= 10:
            c['cot_sum'] += pairwise_official(cot_ne[:10], ds)
            c['cot_n'] += 1

        if len(nv) >= 10:
            c['H_sum'] += -cluster_entropy(nv[:10], dataset=ds)
            c['H_n'] += 1

        pt = _f(p_true_b.get((stem, ds, qid), {}).get('p_true'))
        if pt is None:
            continue
        c['pt_sum'] += pt
        c['pt_n'] += 1

    out = {}
    for k, c in raw.items():
        if not c['n']:
            continue
        out[k] = {
            'op_acc':    c['correct'] / c['n'],
            'op_cons':   c['oc_count'] / c['n'],
            'cot_k10':   c['cot_sum'] / c['cot_n'] if c['cot_n'] else None,
            'neg_H_k10': c['H_sum'] / c['H_n'] if c['H_n'] else None,
            'p_true':    c['pt_sum'] / c['pt_n'] if c['pt_n'] else None,
        }
    return out


def render(cells, out_pdf):
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 9, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        # Embed TrueType rather than Type 3: Type 3 renders poorly when zoomed
        # and is flagged by some archives.
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    nrows = len(ROWS)
    fig, axes = plt.subplots(nrows, 4, figsize=(12, 2.0 * nrows), sharex='col')

    for col, ds in enumerate(NT_DATASETS):
        for row, (ykey, ylabel, trend_color) in enumerate(ROWS):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(DS_LABELS[ds], fontsize=10, fontweight='bold')
            if row == nrows - 1:
                ax.set_xlabel('Accuracy')

            xs, ys = [], []
            for i, stem in enumerate(NT_MODELS):
                cell = cells.get((stem, ds))
                if not cell:
                    continue
                acc = cell['op_acc']; y = cell.get(ykey)
                if acc is None or y is None:
                    continue
                ax.scatter(acc, y, color=COLORS[i % len(COLORS)], s=36, zorder=3,
                           edgecolors='white', linewidths=0.5,
                           label=MODEL_LABELS[stem] if (row == 0 and col == 0) else None)
                xs.append(acc); ys.append(y)

            if len(xs) >= 3:
                r, p = stats.pearsonr(xs, ys)
                m, b = np.polyfit(xs, ys, 1)
                xr = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 100)
                ax.plot(xr, m * xr + b, color=trend_color, linewidth=1.2,
                        linestyle='--', alpha=0.7, zorder=1)
                # Significance is carried by a light panel tint rather than an
                # asterisk. The tint is neutral grey (de-emphasis, not a
                # significance is low), and 'n.s.' repeats it in text so the
                # encoding survives greyscale printing and CVD readers.
                if p >= 0.05:
                    ax.set_facecolor(NONSIG_FACE)
                ns = '' if p < 0.05 else '  n.s.'
                ax.text(0.04, 0.95, f'$r={r:+.2f}${ns}',
                        transform=ax.transAxes, fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                  edgecolor='0.7', linewidth=0.5, alpha=0.9))

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
            ax.grid(True, alpha=0.25, linewidth=0.4)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    ax_oc_bbox = axes[0, 0].get_position()
    ax_cmp_bbox = axes[1, 0].get_position()
    y_div = (ax_oc_bbox.y0 + ax_cmp_bbox.y1) / 2
    x_left = axes[0, 0].get_position().x0
    x_right = axes[0, -1].get_position().x1
    fig.add_artist(mlines.Line2D([x_left, x_right], [y_div, y_div],
                                 transform=fig.transFigure, color='0.55',
                                 linewidth=0.7, linestyle='-', zorder=10))

    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_pdf.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    RESULTS.mkdir(exist_ok=True)
    render(per_cell(), RESULTS / 'fig1.pdf')
