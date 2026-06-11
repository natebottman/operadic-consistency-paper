"""Teaser figure (Fig 1): leave-one-model-out predicted vs actual accuracy.

Two side-by-side panels: predict held-out model accuracy from (left) OC-rate,
(right) canonical CoT-SC_K=10 agreement-rate (Wang et al. 2023). Linear fit
acc ~ a*signal + b on the other 11 models, then evaluate at the held-out
model. Aesthetic is matched to plot_fig1.py: small dots with white edges,
light grid, dashed reference line.
"""
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from analysis.scoring import score_official, is_correct_official, pairwise_official
from analysis.data_loader import load_nonthinking, NT_DATASETS

mpl.rcParams.update({
    'font.family':     'serif',
    'font.size':       8,
    'axes.labelsize':  8,
    'axes.titlesize':  9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

DS_LABEL = {
    'hotpotqa':   'HotpotQA',
    'musique':    'MuSiQue',
    'strategyqa': 'StrategyQA',
    'drop':       'DROP',
}
DS_COLOR = {
    'hotpotqa':   '#1f77b4',
    'musique':    '#ff7f0e',
    'strategyqa': '#2ca02c',
    'drop':       '#d62728',
}


def per_cell_signals():
    """Two cell sets, one per panel.

    OC panel matches analysis/loo_accuracy.py: every row is included
    (no SC filter), so OC and accuracy denominators are the full per-cell n.
    CoT-SC panel requires the extracted-shorts vector to have >=10 non-empty
    samples so the CoT-SC signal is well-defined; OC and accuracy on this
    panel are recomputed on the same filtered set so the LOO regression is
    consistent within the panel. Uses continuous CoT-SC pairwise agreement
    (matched-axis), not a threshold-binarized rate.
    """
    oc_counts  = defaultdict(lambda: {'n': 0, 'oc': 0, 'correct': 0})
    cot_counts = defaultdict(lambda: {'n': 0, 'cot_sum': 0.0, 'correct': 0})
    for r in load_nonthinking():
        ds = r['dataset']; key = (r['model'], ds)
        direct = r['op_direct']; exp = r['op_expansion']
        nv = r['nv_answers']; cot = r['cot_shorts']
        op_f1 = score_official(direct, exp, ds) if direct.strip() and exp.strip() else 0
        oc_counts[key]['n'] += 1
        if op_f1 >= 0.5: oc_counts[key]['oc'] += 1
        correct = is_correct_official(direct, r['ground_truth'], ds, r['aliases'])
        if correct: oc_counts[key]['correct'] += 1
        cot_ne = [s for s in cot if s.strip()]
        if len(cot_ne) >= 10:
            cot_counts[key]['n'] += 1
            cot_counts[key]['cot_sum'] += pairwise_official(cot_ne[:10], ds)
            if correct: cot_counts[key]['correct'] += 1
    oc_cells = {k: (c['oc']/c['n'], 0.0, c['correct']/c['n'])
                for k, c in oc_counts.items() if c['n']}
    cot_cells = {k: (0.0, c['cot_sum']/c['n'], c['correct']/c['n'])
                 for k, c in cot_counts.items() if c['n']}
    return oc_cells, cot_cells


def loo_predictions(cells, ds, signal_idx):
    pairs = [cells[(s, ds)] for s in {k[0] for k in cells} if (s, ds) in cells]
    sigs = np.array([p[signal_idx] for p in pairs])
    accs = np.array([p[2] for p in pairs])
    n = len(sigs)
    preds = np.empty(n)
    for i in range(n):
        sig_train = np.concatenate([sigs[:i], sigs[i+1:]])
        acc_train = np.concatenate([accs[:i], accs[i+1:]])
        a, b = np.polyfit(sig_train, acc_train, 1)
        preds[i] = a * sigs[i] + b
    return preds, accs


def plot_panel(ax, cells, signal_idx):
    all_pred, all_act = [], []
    for ds in NT_DATASETS:
        preds, accs = loo_predictions(cells, ds, signal_idx)
        ax.scatter(preds, accs,
                   s=18, c=DS_COLOR[ds], marker='o',
                   edgecolor='white', linewidth=0.5,
                   label=DS_LABEL[ds], zorder=3)
        all_pred.extend(preds); all_act.extend(accs)
    return all_pred, all_act


def main(out_pdf):
    oc_cells, cot_cells = per_cell_signals()
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4))

    panels = [
        (axes[0], 0, 'OC', oc_cells),
        (axes[1], 1, 'CoT-SC$_{K=10}$', cot_cells),
    ]

    all_data = []
    for ax, idx, sig, cells in panels:
        all_data.append((ax, idx, sig, plot_panel(ax, cells, idx)))

    lo = min(min(d[3][0] + d[3][1]) for d in all_data) - 0.03
    hi = max(max(d[3][0] + d[3][1]) for d in all_data) + 0.03

    for ax, idx, sig, (preds, acts) in all_data:
        ax.plot([lo, hi], [lo, hi], '--', color='0.4', linewidth=0.9,
                alpha=0.7, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f'Predicted accuracy ({sig})')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.25, linewidth=0.4, zorder=0)
        mae = np.mean(np.abs(np.array(preds) - np.array(acts)))
        ax.text(0.04, 0.96,
                f'MAE = {mae*100:.1f} pp',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='0.7', linewidth=0.5, alpha=0.9))

    axes[0].set_ylabel('Actual accuracy')
    axes[1].set_yticklabels([])

    plt.tight_layout(pad=0.3, rect=[0.22, 0, 1, 1], w_pad=4.0)
    # Legend at vertical midpoint of the panels, in the left strip.
    handles, labels = axes[0].get_legend_handles_labels()
    pos = axes[0].get_position()
    mid_y = 0.5 * (pos.y0 + pos.y1)
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(0.0, mid_y), frameon=False,
               fontsize=7.5, handletextpad=0.4, borderaxespad=0.3,
               borderpad=0.3)
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'teaser.pdf')
    main(out)
