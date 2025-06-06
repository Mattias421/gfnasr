import random
import subprocess
import torch
from torch.distributions import Categorical

from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.utils.data_utils import undo_padding
from speechbrain.decoders.utils import _update_mem
import heapq  # For keeping the buffer as a min-heap (to easily pop smallest rewards)
import editdistance  # For similarity check
import gzip
import pickle
import numpy as np  # For random sampling


def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    reward_weight,
    subtb_lambda=1.0,
):
    print(log_pf.shape)
    print(log_r.shape)
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1]
    )
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    delta = (
        log_r[:, :-1] * reward_weight
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:] * reward_weight
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_text
    mask = (generated_text[:, :-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1]
    for subtraj_len in range(1, generated_len):
        subtb_term = (
            delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        ) ** 2
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += (
            subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        )
    batch_loss /= total_lambda

    return batch_loss


class GFNPolicy(S2SWhisperGreedySearcher):
    def __init__(
        self,
        model,
        temp_high=1.0,
        temp_low=0.5,
        temp_prob=0.666,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.temp_high = temp_high
        self.temp_low = temp_low
        self.temp_prob = temp_prob

    def forward(
        self,
        gfn_model,
        enc_states,
        wav_len,
        target_words,
        temperature=None,
        action_seq=None,
        skip_reward=False,
        skip_first=4,
        ref_token_ids=None,
    ):

        if random.random() < self.temp_prob and temperature is None:  # With tempering
            temperature = (
                random.random() * (self.temp_high - self.temp_low) + self.temp_low
            )
        elif temperature is None:
            temperature = 1.0

        # generate and return the probability of terminating at every step
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]
        active_seqs = enc_states.new_ones(batch_size).bool()

        state = self.reset_mem(batch_size, device=device)

        # Using bos as the first input
        token_ids = (
            enc_states.new_zeros(batch_size).fill_(self.bos_index).long()
        ).unsqueeze(-1)

        min_len = int(enc_states.shape[1] * self.min_decode_ratio)
        max_len = int(enc_states.shape[1] * self.max_decode_ratio)

        log_pf = []
        log_pterm = []

        for i in range(max_len + 1):
            if action_seq is None:
                # generate action_seq from policy
                logits, modified_logits, state, _ = self.forward_step(
                    gfn_model, token_ids.squeeze(-1), state, enc_states, enc_lens
                )

                with torch.no_grad():
                    prob = logits.softmax(dim=-1)

                    if i < min_len:
                        # if we haven't reach the minimum length, set the probability of terminating to 0
                        modified_logits[:, self.eos_index] = -torch.inf
                    elif i >= max_len:
                        # if we've reached the maximum length, set the probability of terminating to 1
                        mask = [True] * modified_logits.shape[1]
                        mask[self.eos_index] = False
                        modified_logits[:, mask] = -torch.inf

                    if temperature != 1.0:
                        prob = (modified_logits / temperature).softmax(dim=-1)
                        token_ids = torch.multinomial(prob, num_samples=1)
                    else:
                        token_ids = modified_logits.argmax(dim=-1)[:, None]
            else:
                # use action seq from buffer
                # TODO perhaps this could be taken outside of loop and have logits computed at once for faster training?
                state = _update_mem(token_ids.squeeze(-1), state)
                logits, attn, kv = gfn_model.forward_decoder(
                    enc_states, state, past_key_values=self.kv_cache
                )
                logits = logits[:,-1]
                if i >= action_seq.size(-1):
                    token_ids = (
                        torch.ones_like(action_seq[:, 0]) * self.eos_index
                    ).unsqueeze(-1)
                else:
                    token_ids = action_seq[:, i].unsqueeze(-1)

            token_ids = torch.where(
                active_seqs.unsqueeze(-1),
                token_ids,
                self.eos_index,
            )
            logprob = logits.log_softmax(dim=-1)
            log_pterm.append(
                torch.where(
                    active_seqs,
                    logprob[:, self.eos_index],
                    0,
                )
            )
            active_seqs = active_seqs * (token_ids != self.eos_index).squeeze(-1)
            log_pf.append(
                torch.where(
                    active_seqs,
                    logprob.gather(-1, token_ids).squeeze(-1),
                    0,
                )
            )
            # check if all sequences have terminated
            if torch.all(~active_seqs) or self._check_end_condition(state):
                break

        # compute reward

        log_pf = torch.stack(log_pf, dim=1)
        log_pterm = torch.stack(log_pterm, dim=1)

        normalized_fn = self.tokenizer.normalize

        target_words = [normalized_fn(text).split(" ") for text in target_words]

        # add termination token
        state = torch.cat([state[:, skip_first:], token_ids], dim=-1)

        log_r = torch.zeros_like(log_pf)
        max_len = log_r.shape[-1]
        for i in range(max_len):
            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in state[:, :i+1]
            ]
            predicted_words = [
                normalized_fn(text).split(" ") for text in predicted_words
            ]
            for j, sentence in enumerate(predicted_words):
                if state[j,i] == self.eos_index and state[j,i-1] == self.eos_index:
                    log_r[j,i] = -1
                else:
                    log_r[j,i] = - editdistance.eval(' '.join(target_words[j]), ' '.join(sentence)) / len(' '.join(target_words[j]))



        return state, log_pf, log_pterm, log_r

    def forward_step(self, gfn_model, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        tokens = _update_mem(inp_tokens, memory)

        logits, attn, kv = gfn_model.forward_decoder(
            enc_states, tokens, past_key_values=self.kv_cache
        )

        if tokens.shape[1] == self.sample_begin:
            probs_at_bos = (
                logits[:, self.initial_tokens.index(self.model.bos)]
                .float()
                .softmax(dim=-1)
            )
            self.no_speech_probs = probs_at_bos[:, self.model.no_speech].tolist()

        logits = logits[:, -1]
        modified_logits = logits.clone().detach()

        if self.use_kv_cache:
            self.kv_cache = kv

        if self.suppress_blank:
            if tokens.shape[1] == self.sample_begin:
                modified_logits[
                    :,
                    self.model.tokenizer.encode(" ", add_special_tokens=False)
                    + [self.eos_index],
                ] = -torch.inf

        if self.suppress_tokens:
            if self.model.config.suppress_tokens is None:
                tokens_to_suppress = self.get_tokens_to_suppress
            else:
                tokens_to_suppress = self.model.get_suppress_tokens

            modified_logits[:, list(tokens_to_suppress)] = -torch.inf

        return logits, modified_logits, tokens, attn


class ReplayBuffer:
    """
    A replay buffer that uses a heap to keep the max_size items with the highest reward
    for each unique audio utterance.
    """

    def __init__(self, buffer_size, sim_tolerance=0.25):
        self.buffer_size_per_utterance = buffer_size
        self.termination_token_id = None
        self.tokenizer = None
        self.sim_tolerance = sim_tolerance
        self._buffer = {}  # Key: utt_id, Value: {"sentences": min_heap, "exists_str": set_of_str_sentences}

    def reset(self):
        self._buffer = {}

    def _get_clean_token_list(self, tensor_sentence):
        """Helper to get a list of tokens excluding termination_token_id."""
        return [
            token.item()
            for token in tensor_sentence
            if token.item() != self.termination_token_id
        ]

    def add(
        self, utt_id: str, tensor_sentence: torch.Tensor, full_logrewards: torch.Tensor
    ):
        """
        Add a single generated sentence for a given utterance ID to the buffer.

        Args:
            utt_id (str): Unique identifier for the audio utterance.
            tensor_sentence (torch.Tensor): Tensor of shape (seq_len,) representing the generated tokens.
            full_logrewards (torch.Tensor): Tensor of shape (seq_len,) representing log rewards at each step.
        """
        if utt_id not in self._buffer:
            self._buffer[utt_id] = {
                "sentences": [],  # This will be a min-heap: (final_log_reward, str_sentence, tensor_sentence, full_logrewards)
                "exists_str": set(),  # Set of string representations of sentences already in buffer
            }

        # Determine the actual length before padding/EOS and get final log reward
        actual_len = (tensor_sentence != self.termination_token_id).sum().item()
        if actual_len == 0:  # Empty sentence, or all EOS
            final_log_reward = -float("inf")  # Or some other very small number
        else:
            # Reward for the state *before* the EOS token, or the last non-EOS token
            # log_rewards are for s_0, s_1, ..., s_T-1, s_T (EOS)
            # if actual_len corresponds to the index of EOS, we want reward at actual_len-1
            # if actual_len corresponds to the last non-EOS token (if no EOS), reward at actual_len-1
            reward_idx = (
                min(actual_len - 1, len(full_logrewards) - 1) if actual_len > 0 else 0
            )
            if reward_idx < 0:
                reward_idx = 0  # handle case where actual_len is 0 after min

            if reward_idx < len(full_logrewards):
                final_log_reward = full_logrewards[reward_idx].item()
            else:  # Should not happen if full_logrewards corresponds to tensor_sentence length
                final_log_reward = -float("inf")

        # Decode to string for deduplication and similarity check (excluding special tokens)
        # Ensure tensor_sentence is on CPU and is a 1D tensor of integers for the tokenizer
        clean_tokens_for_decode = tensor_sentence[:actual_len].long().cpu()
        str_sentence = self.tokenizer.decode(
            clean_tokens_for_decode, skip_special_tokens=True
        ).strip()

        if not str_sentence:  # Skip empty strings after decoding
            return

        if str_sentence in self._buffer[utt_id]["exists_str"]:
            return  # Exact string duplicate

        tokenized_sentence_list = self._get_clean_token_list(
            tensor_sentence[:actual_len]
        )

        # Similarity check and potential replacement
        # Need to iterate carefully as we might modify the heap
        current_heap = self._buffer[utt_id]["sentences"]
        item_to_add = (
            final_log_reward,
            str_sentence,
            tensor_sentence.cpu(),
            full_logrewards.cpu(),
        )  # Store on CPU

        replaced_similar = False
        for i in range(
            len(current_heap) - 1, -1, -1
        ):  # Iterate backwards for safe removal
            buffer_item_reward, buffer_str_sent, buffer_tensor_sent, _ = current_heap[i]
            tokenized_existing_list = self._get_clean_token_list(buffer_tensor_sent)

            # Calculate edit distance based on token lists for more robustness
            # Using a relative threshold
            edit_dist = editdistance.eval(
                tokenized_sentence_list, tokenized_existing_list
            )
            combined_len_for_sim = len(tokenized_sentence_list) + len(
                tokenized_existing_list
            )
            similarity_threshold_val = (
                combined_len_for_sim * self.sim_tolerance
                if combined_len_for_sim > 0
                else 0
            )

            if edit_dist < similarity_threshold_val:
                if buffer_item_reward >= final_log_reward:
                    return  # Existing similar item is better or equal, do nothing
                else:
                    # New item is better, remove old one and add new one
                    self._buffer[utt_id]["exists_str"].remove(buffer_str_sent)
                    current_heap.pop(i)  # Remove by index
                    # No need to re-heapify immediately if we add later
                    replaced_similar = True
                    break  # Found a similar item and handled it

        # Add the new item
        heap_for_utt = self._buffer[utt_id]["sentences"]
        str_set_for_utt = self._buffer[utt_id]["exists_str"]

        if len(heap_for_utt) >= self.buffer_size_per_utterance:
            if (
                final_log_reward > heap_for_utt[0][0]
            ):  # Only add if better than the worst in buffer
                popped_reward, popped_str, _, _ = heapq.heappushpop(
                    heap_for_utt, item_to_add
                )
                str_set_for_utt.remove(popped_str)
                str_set_for_utt.add(str_sentence)
        else:
            heapq.heappush(heap_for_utt, item_to_add)
            str_set_for_utt.add(str_sentence)


    def add_batch(
        self,
        utt_ids: list,
        generated_sentences: torch.Tensor,
        full_logrewards_batch: torch.Tensor,
    ):
        """
        Add a batch of items to the buffer.

        Args:
            utt_ids (list[str]): List of utterance IDs for the batch.
            generated_sentences (torch.Tensor): Batch of generated sentences (B, T_gen).
            full_logrewards_batch (torch.Tensor): Batch of log_rewards (B, T_gen).
        """
        # Ensure generated_sentences and log_rewards are on CPU for storage/processing
        # generated_sentences = generated_sentences.cpu()
        # full_logrewards_batch = full_logrewards_batch.cpu()

        for i in range(generated_sentences.size(0)):
            utt_id = utt_ids[i]
            tensor_sentence = generated_sentences[i]
            logrewards_for_sentence = full_logrewards_batch[i]

            self.add(utt_id, tensor_sentence, logrewards_for_sentence)

    def sample(
        self, batch_size: int, current_batch_utt_ids: list, device: torch.device
    ):
        """
        Sample a batch of items, prioritizing those from the current batch's utterances.
        Returns padded tensor_sentences and full_logrewards.

        Args:
            batch_size (int): Desired batch size for the sample.
            current_batch_utt_ids (list[str]): Utterance IDs from the current training batch.
            device (torch.device): Device to put the sampled tensors on.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] or Tuple[None, None]:
                Padded sentences and corresponding log rewards, or None if not enough data.
        """
        sampled_sentences = []
        sampled_log_rewards = []

        # Collect all available items from relevant utterance IDs
        available_items = []
        for utt_id in current_batch_utt_ids:
            if utt_id in self._buffer and self._buffer[utt_id]["sentences"]:
                # _buffer stores (final_log_reward, str_sentence, tensor_sentence, full_logrewards)
                # We need tensor_sentence and full_logrewards for training
                for item in self._buffer[utt_id]["sentences"]:
                    available_items.append(
                        (item[2], item[3])
                    )  # (tensor_sentence, full_logrewards)

        if not available_items:
            return None, None

        # Sample with replacement if not enough unique items
        num_to_sample = (
            min(batch_size, len(available_items))
            if batch_size < len(available_items)
            else batch_size
        )
        replace_sample = num_to_sample > len(available_items)

        indices = np.random.choice(
            len(available_items),
            size=num_to_sample,
            replace=replace_sample,  # Sample with replacement if batch_size > available items
        )

        for idx in indices:
            sampled_sentences.append(available_items[idx][0])
            sampled_log_rewards.append(available_items[idx][1])

        if not sampled_sentences:
            return None, None

        # Pad the sampled tensors
        padded_sentences = torch.nn.utils.rnn.pad_sequence(
            sampled_sentences,
            batch_first=True,
            padding_value=self.termination_token_id,
        ).to(device)

        padded_log_rewards = torch.nn.utils.rnn.pad_sequence(
            sampled_log_rewards,
            batch_first=True,
            padding_value=0.0,  # Assuming 0 is a safe padding for log rewards
        ).to(device)

        return padded_sentences, padded_log_rewards

    def __len__(self):
        """Total number of items across all utterance buffers."""
        return sum(len(data["sentences"]) for data in self._buffer.values())

    def get_utterance_buffer_size(self, utt_id):
        if utt_id in self._buffer:
            return len(self._buffer[utt_id]["sentences"])
        return 0


def buffer_save(obj, path):
    # Ensure all tensors in buffer are on CPU before pickling
    cpu_buffer = {}
    for utt_id, data in obj._buffer.items():
        cpu_buffer[utt_id] = {
            "sentences": [
                (r, s_str, ts.cpu(), fr.cpu()) for r, s_str, ts, fr in data["sentences"]
            ],
            "exists_str": data["exists_str"],
        }
    with gzip.open(path, "wb") as f:
        pickle.dump(cpu_buffer, f)

def buffer_load(obj, path, epoch):
    try:
        with gzip.open(path, "rb") as f:
            obj._buffer = pickle.load(f)
        # Ensure heaps are valid after loading (though pickle should preserve structure)
        for utt_id in obj._buffer:
            heapq.heapify(obj._buffer[utt_id]["sentences"])
    except FileNotFoundError:
        print(f"No replay buffer found at {path}. Starting with an empty buffer.")
        obj.reset()
    except Exception as e:
        print(f"Error loading replay buffer: {e}. Starting with an empty buffer.")
        obj.reset()


# def tempered_lm_reward():
#     log_pf = torch.stack(log_pf, dim=1)
#     log_pterm = torch.stack(log_pterm, dim=1)
#
#     if skip_reward:
#         log_r, log_r_unpenalized = None, None
#     else:
#         self.model.cpu()
#         self.reward_model.to(state.device)
#         with torch.no_grad():
#             logits, _, _ = self.reward_model.forward_decoder(enc_states, state)
#             # get rid of the first few tokens
#             logits = logits[:, skip_first - 1 :]
#             # score the log probability of the input sequence while ignoring termination and padding tokens
#             logprob = logits.log_softmax(-1)
#             reward_token_ids = state[:, skip_first:].unsqueeze(-1)
#             logPF = logprob[:, :-1].gather(-1, reward_token_ids).squeeze(-1)
#             logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)
#             reward = logprob[
#                 :, :, self.eos_index
#             ]  # logP(generated[i+1]=term | prompt + generated[:i+1])
#             reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
#             non_term_mask = (state != self.eos_index)[:, skip_first:]
#             non_term_mask = torch.cat(
#                 (
#                     non_term_mask.new_ones(non_term_mask.shape[0], 1),
#                     non_term_mask,
#                 ),
#                 dim=-1,
#             )  # Start (i.e., empty) state has never terminated
#             reward[~non_term_mask] = 0.0
#             log_r_unpenalized = reward.clone()
#             log_r = torch.where(
#                 non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward
#             )
#
#         self.reward_model.cpu()
#         self.model.to(state.device)
