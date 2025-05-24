#!/usr/bin/env python3
"""Recipe for training a whisper-based ASR system with librispeech.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take the whisper encoder-decoder to fine-tune on the NLL.

If you want to only use the whisper encoder system, please refer to the recipe
speechbrain/recipes/LibriSpeech/ASR/CTC/train_with_whisper.py

To run this recipe, do the following:
> python train_with_whisper.py hparams/train_hf_whisper.yaml

To add adapters and train only a fraction of the parameters, do:
> python train_with_whisper.py hparams/train_whisper_lora.yaml

Authors
 * Peter Plantinga 2024
 * Adel Moumen 2022, 2024
 * Titouan Parcollet 2022
"""

import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

from dataset import HDF5Dataset
from utils import get_sclite_wer

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        embeds, utt_id, wav_lens = batch
        embeds = embeds.to(self.device)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(
                    embeds, wav_lens / wav_lens.max()
            )
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.policy(embeds, wav_lens / wav_lens.max())

        return hyps, utt_id



    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        (hyps, utt_id) = predictions

        if stage != sb.Stage.TRAIN:
            # tokens, tokens_lens = batch.tokens

            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in hyps
            ]

            # Convert indices to words
            if hasattr(self.hparams, "normalized_transcripts"):

                if hasattr(self.tokenizer, "normalize"):
                    normalized_fn = self.tokenizer.normalize
                else:
                    normalized_fn = self.tokenizer._normalize

                predicted_words = [
                        normalized_fn(text).upper() for text in predicted_words
                ]

            else:
                predicted_words = [text for text in predicted_words]

            for pred, i in zip(predicted_words, utt_id):
                self.hypothesis.append(f"{pred} ({i})\n")


        
        return torch.tensor([0])

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.hypothesis = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            if epoch is not None:
                hyps_file = Path(self.hparams.output_folder) / "valid_hyps_{epoch}.trn"
                with open(hyps_file, 'w') as f:
                    f.writelines(self.hypothesis)

                stage_stats["WER"] = get_sclite_wer(Path("./data/valid.trn"), hyps_file)
                stage_stats["CER"] = get_sclite_wer(Path("./data/valid.trn"), hyps_file, cer=True)

            else:
                hyps_file = Path(self.hparams.output_folder) / "test_hyps.trn"

                with open(hyps_file, 'w') as f:
                    f.writelines(self.hypothesis)

                stage_stats["WER"] = get_sclite_wer(Path("./data/test.trn"), hyps_file)
                stage_stats["CER"] = get_sclite_wer(Path("./data/test.trn"), hyps_file, cer=True)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing_whisper.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)




if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    train_data = HDF5Dataset(hparams["train_data_path"])
    valid_data = HDF5Dataset(hparams["valid_data_path"])
    test_data = HDF5Dataset(hparams["test_data_path"])

    modules = hparams["modules"]

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    # Training
    # asr_brain.fit(
    #     asr_brain.hparams.epoch_counter,
    #     train_data,
    #     valid_data,
    #     train_loader_kwargs=hparams["train_loader_kwargs"],
    #     valid_loader_kwargs=hparams["valid_loader_kwargs"],
    # )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    asr_brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_loader_kwargs"],
        min_key="WER",
    )
