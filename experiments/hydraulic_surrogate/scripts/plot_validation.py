"""Produce a compact comparison from recorded validation manifests."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', default='artifacts/validation_comparison.png')
    args=parser.parse_args()
    manifests={r:json.loads((ROOT/'artifacts'/r/'manifest.json').read_text()) for r in args.runs}
    fig, axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for r,m in manifests.items():
        years=list(m['folds'])
        axes[0].plot(years,[m['folds'][y]['micro_auc'] for y in years],marker='o',label=r)
    axes[0].set(title='Purged chronological validation',ylabel='Micro ROC AUC',xlabel='Held-out year')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    latest=manifests[args.runs[-1]]
    for year,fold in latest['folds'].items():
        units=list(fold['unit_auc'])
        axes[1].plot([x.replace('_G',' ') for x in units],list(fold['unit_auc'].values()),marker='.',label=year)
    axes[1].set(title=f'Per-unit ranking: {args.runs[-1]}',ylabel='ROC AUC')
    axes[1].tick_params(axis='x',rotation=75)
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.savefig(ROOT/args.output,dpi=160)

if __name__=='__main__':
    main()
