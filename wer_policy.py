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

import random

from speechbrain.utils.edit_distance import wer_details_for_batch

from dataset import HDF5Dataset

logger = get_logger(__name__)



# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        embeds, utt_id, wav_lens, refs = batch


        def _generate_word(context_word):
            while True:
                # Generate returns a list, so we take the first element
                generated_word = self.bigram_lm.generate(1, text_seed=[context_word])
                if generated_word != '</s>':
                    return generated_word

        states = []

        for ref in refs:
            n_err = 0
            stack = ref.split()
            N = len(stack)
            state = []

            while stack:
                logits = - torch.tensor([n_err, n_err+1, n_err+1, n_err+1]) / N
                prob = (logits / self.hparams.policy_temp_low).softmax(dim=-1)

                match torch.multinomial(prob, num_samples=1):
                    case 0:
                        # correct
                        state.append(stack.pop(0))
                    case 1:
                        # delete
                        stack.pop(0)
                    case 2:
                        # insert
                        if state == []:
                            state.append(_generate_word('</s>'))
                        else:
                            state.append(_generate_word(state[-1]))
                    case 3:
                        # substitute
                        if state == []:
                            state.append(_generate_word('</s>'))
                            stack.pop(0)
                        else:
                            state.append(_generate_word(state[-1]))
                            stack.pop(0)

            states.append(state)

        policy_outputs = [self.tokenizer.normalize(' '.join(text)) for text in states]
        policy_tokens = [self.tokenizer.encode(text)[4:] for text in policy_outputs]

        max_len = max(len(seq) for seq in policy_tokens)
        padded_sequences = []
        for seq in policy_tokens:
            num_padding = max_len - len(seq)
            # Create a new list with the original sequence + padding
            padded_seq = seq + [self.hparams.policy.eos_index] * num_padding
            padded_sequences.append(padded_seq)

        action_seq = torch.tensor(padded_sequences).to(self.device)

        embeds = embeds.to(self.device)

        skip_reward = False  # (stage == sb.Stage.TEST)

        state, log_probs, log_probs_term, log_reward = (
            self.hparams.policy(
                self.modules.whisper,
                embeds,
                wav_lens / wav_lens.max(),
                target_words=refs,
                temperature=1.0,
                action_seq=action_seq,
                skip_reward=skip_reward,
            )
        )
        breakpoint()

        # if random.random() < self.hparams.use_buffer_prob and self.hparams.replay_buffer.sample(len(utt_id), list(utt_id), self.device)[0] is not None:
        #
        #     action_seq, log_r = self.hparams.replay_buffer.sample(
        #         len(utt_id), list(utt_id), self.device
        #     )
        #     state, log_probs, log_probs_term, log_reward = (
        #         self.hparams.policy(
        #             self.modules.whisper,
        #             embeds,
        #             wav_lens / wav_lens.max(),
        #             target_words=refs,
        #             temperature=1.0,
        #             action_seq=action_seq,
        #             skip_reward=skip_reward,
        #         )
        #     )
        # else:
        #     state, log_probs, log_probs_term, log_reward = (
        #         self.hparams.policy(
        #             self.modules.whisper,
        #             embeds,
        #             wav_lens / wav_lens.max(),
        #             target_words=refs,
        #             temperature=temperature,
        #             skip_reward=skip_reward,
        #         )
        #     )
        #
        #     if stage == sb.Stage.TRAIN:
        #         self.hparams.replay_buffer.add_batch(
        #             utt_ids=utt_id, generated_sentences=state, full_logrewards_batch=log_reward
        #         )

        if stage == sb.Stage.VALID:
            hyps, lengths, scores, model_log_probs = self.hparams.beam_search(embeds, wav_lens)
        else:
            scores, hyps = None, None

        return (
            utt_id,
            state,
            log_probs,
            log_probs_term,
            log_reward,
            hyps,
            scores,
        )



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


    del hparams["whisper"].adapted_model.model.encoder

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
    asr_brain.hparams.policy.tokenizer = tokenizer

    # get train lm
    from nltk.lm import MLE
    from nltk.lm.preprocessing import padded_everygram_pipeline
    train_txt, vocab = padded_everygram_pipeline(2, [x[-1].split(' ') for x in train_data])
    asr_brain.bigram_lm = MLE(2)
    asr_brain.bigram_lm.fit(train_txt, vocab)


    hparams["replay_buffer"].termination_token_id = hparams["policy"].eos_index
    hparams["replay_buffer"].tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )

    # Testing
    if hparams["evaluate"]:
        os.makedirs(hparams["output_wer_folder"], exist_ok=True)

        asr_brain.evaluate(
            test_data,
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="WER",
        )
