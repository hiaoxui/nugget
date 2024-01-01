import os
from typing import *
import json
import shutil
import multiprocessing

from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl

from nugget.utils.types import Nuggets, NuggetInspect
from nugget.inspect.highlight import gen_highlight
from nugget.inspect.composition_plot import composition_plot
from dlutils.tokenizer_wihtout_warning import load_tokenizer


def collection_and_plot(inspect_path: str, pretrained: str):
    # Collect
    cache_path = os.path.join(inspect_path, 'cache')
    nuggets = list()
    lines = list()
    for fn in sorted(os.listdir(cache_path)):
        for line in open(os.path.join(cache_path, fn)):
            nuggets.append(json.loads(line))
            lines.append(line.strip())
    with open(os.path.join(inspect_path, 'nuggets.jsonl'), 'w') as fp:
        fp.write('\n'.join(lines))
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, True)
    nuggets = [NuggetInspect(nug['tokens'], nug['index'], nug['scores']) for nug in nuggets]

    # Convert to tokens
    tok = load_tokenizer(pretrained, use_fast=False)
    for nug in nuggets:
        nug.to_tokens(tok)

    # Plot
    gen_highlight(nuggets[:128], os.path.join(inspect_path, 'highlight.html'))
    composition_plot(nuggets, os.path.join(inspect_path, 'composition.jpg'))


class InspectCallback(Callback):
    """
    Dump nuggets to jsonl files and plot figs to analyze them.
    Workers dump this jsonl files to `cache` folder, and the rank_zero worker collects them
    and plot the figs.
    """
    def __init__(self, period: int, cache: str, pretrained: str, debug: bool = False):
        self.period, self.cache, self.pretrained = period, cache, pretrained
        self.debug = debug
        self.current_idx = 0
        self.val_outputs: List[NuggetInspect] = list()
        self.save_path = None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_path = os.path.join(
            self.cache, 'inspect', f'step_{trainer.global_step:06}', 'cache', f'{trainer.global_rank}.jsonl'
        )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if hasattr(trainer, 'freeze_scorer') and trainer.freeze_scorer:
            return
        if self.period <= 0 or (self.current_idx % self.period != 1 and self.period != 1):
            return
        # assume the first output is nugget
        nuggets: Nuggets = outputs[0]
        for i in range(nuggets.encoding.shape[0]):
            self.val_outputs.append(NuggetInspect(
                batch['input_ids'][i][batch['attention_mask'][i]].tolist(),
                nuggets.index[i][nuggets.mask[i]].tolist(),
                nuggets.scores[i][nuggets.mask[i]].tolist(),
            ))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.val_outputs:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as fp:
                fp.write('\n'.join(map(json.dumps, map(vars, self.val_outputs))))

        inspect_path = os.path.dirname(os.path.dirname(self.save_path))
        if trainer.is_global_zero and os.path.exists(inspect_path):
            if not self.debug:
                multiprocessing.Process(
                    None, collection_and_plot, args=(inspect_path, self.pretrained)
                ).start()
            else:
                collection_and_plot(inspect_path, self.pretrained)

        self.val_outputs = list()
        self.save_path = None
        self.current_idx += 1
