from typing import *
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np

from nugget.utils.types import NuggetInspect


def composition_plot(outs: List[NuggetInspect], save_path: str):
    max_seq_length = max(len(nug.tokens) for nug in outs)
    nugget_idx_cnt = np.zeros([max_seq_length], np.int64)
    token_idx_cnt = np.zeros([max_seq_length], np.int64)
    token_cnt = defaultdict(int)
    n_nugget = 0
    for nug in outs:
        for ni in nug.index:
            nugget_idx_cnt[ni] += 1
            token_cnt[nug.tokens[ni]] += 1
        n_nugget += len(nug.index)
        token_idx_cnt[:len(nug.tokens)] += 1
    token_cnt = sorted(list(token_cnt.items()), key=lambda z: -z[1])
    nugget_pct = nugget_idx_cnt / (token_idx_cnt.astype(np.float64) + 1.) * 100
    with plt.style.context('ggplot'):
        fig, (nugget_dist, pie) = plt.subplots(1, 2, figsize=(10, 4))
        nugget_dist.plot(np.arange(max_seq_length), nugget_pct)
        nugget_dist.set_xlabel('token index')
        nugget_dist.set_ylabel('select prob*100')

        n_token_pie = 20
        pie_labels, pie_counts = map(list, zip(*token_cnt[:n_token_pie]))
        pie_counts.append(n_nugget - sum(pie_counts))
        pie_labels.append('others')
        pie.pie(pie_counts, labels=pie_labels)
        fig.tight_layout()
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
