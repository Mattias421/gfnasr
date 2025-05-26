import subprocess
import torch
from torch.distributions import Categorical

from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.utils.data_utils import undo_padding
from speechbrain.decoders.utils import (
        _update_mem)

def get_sclite_wer(ref_trn_path, hyp_trn_path, cer=False):
    """
    Runs sclite to calculate WER and extracts only the WER percentage.

    Args:
        ref_trn_path (str): Path to the reference .trn file.
        hyp_trn_path (str): Path to the hypothesis .trn file.

    Returns:
        float: The Word Error Rate (WER) as a percentage, or None if an error occurs
               or WER cannot be parsed.
    """
    sclite_command_parts = [
        "sclite",
        "-r", ref_trn_path,
        "-h", hyp_trn_path,
        "-i", "spu_id",
        "-o", "sum", "stdout" # 'pralign' or other report types can be added if needed by sclite
                              # but for just getting WER from summary, 'sum stdout' is often enough.
    ]

    if cer:
        sclite_command_parts.append("-c")

    pipeline_command = (
        f"{' '.join(sclite_command_parts)} "
        f"| grep 'Sum/Avg' " # This grep might need to be more specific, e.g., grep '^| Sum/Avg'
        f"| awk '{{print $10}}'" # Using $10 as per your original script.
                               # Note the double curly braces for awk in an f-string.
    )

    try:
        result = subprocess.run(pipeline_command, shell=True, capture_output=True, text=True, check=True)

        # stdout will contain the output of the last command in the pipe (awk)
        wer_str = result.stdout.strip()

        if wer_str:
            return float(wer_str)
        else:
            print("Warning: WER string is empty. sclite output might not have matched.")
            print("sclite stderr:", result.stderr)
            return None

    except subprocess.CalledProcessError as e:
        # This exception is raised if the command returns a non-zero exit code
        print(f"Error running sclite pipeline: {e}")
        return None
    except FileNotFoundError:
        print("Error: sclite command not found. Is it in your PATH?")
        return None
    except ValueError:
        print(f"Error: Could not convert extracted WER '{wer_str}' to float.")
        return None


def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1] 
    )
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    breakpoint()

    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:]
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_text
    mask = (generated_text[:, :] == termination_token_id).cumsum(-1) >= 1

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
    def __init__(self, model, reward_model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.reward_model = reward_model

    def forward(
        self,
        enc_states,
        wav_len,
        max_len=10,
        min_len=0,
        temperature=1.0,
        action_seq=None,
        skip_reward=False,
        skip_first=4,
    ):
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
            logits, modified_logits, state, _ = self.forward_step(
                    token_ids.squeeze(-1), state, enc_states, enc_lens
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

                prob = (modified_logits / temperature).softmax(dim=-1)
                token_ids = torch.multinomial(prob, num_samples=1)

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
            if torch.all(~active_seqs):
                break

        if skip_reward:
            log_r, log_r_unpenalized = None, None
        else:
            log_pf = torch.stack(log_pf, dim=1)
            log_pterm = torch.stack(log_pterm, dim=1)

            logits, _, _ = self.reward_model.forward_decoder(enc_states, state)
            # get rid of the first few tokens
            logits = logits[:, skip_first - 1 :]
            # score the log probability of the input sequence while ignoring termination and padding tokens
            logprob = logits.log_softmax(-1)
            reward_token_ids = state[:, skip_first:].unsqueeze(-1)
            logPF = logprob[:, :-1].gather(-1, reward_token_ids).squeeze(-1)
            logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)
            reward = logprob[
                :, :, self.eos_index
            ]  # logP(generated[i+1]=term | prompt + generated[:i+1])
            reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
            non_term_mask = (state != self.eos_index)[:, skip_first:]
            non_term_mask = torch.cat(
                (
                    non_term_mask.new_ones(non_term_mask.shape[0], 1),
                    non_term_mask,
                ),
                dim=-1,
            )  # Start (i.e., empty) state has never terminated
            reward[~non_term_mask] = 0.0
            log_r_unpenalized = reward.clone()
            log_r = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)

        # add termination token 
        state = torch.cat([state[:, skip_first:], token_ids], dim=-1)
        breakpoint()
        return state, log_pf, log_pterm, log_r, log_r_unpenalized

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
            """Performs a step in the implemented beamsearcher."""
            tokens = _update_mem(inp_tokens, memory)

            logits, attn, kv = self.model.forward_decoder(
                enc_states, tokens, past_key_values=self.kv_cache
            )


            if tokens.shape[1] == self.sample_begin:
                probs_at_bos = (
                    logits[:, self.initial_tokens.index(self.model.bos)]
                    .float()
                    .softmax(dim=-1)
                )
                self.no_speech_probs = probs_at_bos[
                    :, self.model.no_speech
                ].tolist()

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


    def forward_old(self, enc_states, wav_len, skip_reward=False):
        """This method performs a greedy search.

        Arguments
        ---------
        enc_states : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        wav_len : torch.Tensor
            The speechbrain-style relative length.

        Returns
        -------
        hyps : List[List[int]]
            List containing the hypotheses.
        top_lengths : torch.Tensor (batch)
            This tensor contains the length of each hypothesis.
        top_scores : torch.Tensor (batch)
            The score of each hypotheses.
        top_log_probs : torch.Tensor (batch, max length of token_id sequences)
            The log probabilities of each hypotheses.
        """
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        # memory acts as state here
        memory = self.reset_mem(batch_size, device=device)

        # Using bos as the first input
        inp_tokens = (
            enc_states.new_zeros(batch_size).fill_(self.bos_index).long()
        )

        log_probs_lst = []
        log_pf = []
        min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        min_decode_steps, max_decode_steps = self.change_max_decoding_length(
            min_decode_steps, max_decode_steps
        )

        has_ended = enc_states.new_zeros(batch_size).bool()
        for step in range(min_decode_steps, max_decode_steps):
            logits, memory, _ = self.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )

            inp_tokens = Categorical(
                logits=logits / self.temperature
            ).sample()

            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            log_probs_lst.append(log_probs)
            log_pf.append(
            torch.where(
                ~has_ended,
                log_probs.gather(-1, inp_tokens[:,None]).squeeze(-1),
                0,))


            has_ended = has_ended | (inp_tokens == self.eos_index)
            log_probs[has_ended] = -torch.inf
            inp_tokens[has_ended] = self.eos_index

            if has_ended.all() or self._check_end_condition(memory):
                break

        log_probs = torch.stack(log_probs_lst, dim=1)
        log_probs_term = log_probs[:, :, self.eos_index]
        log_pf = torch.stack(log_pf, dim=1)

        if skip_reward:
            log_reward, log_reward_unpenalized = None, None

        hyps = torch.cat([memory, inp_tokens[:, None]], dim=-1)

        reward_logits, _, _ = self.reward_model.forward_decoder(enc_states, hyps)

        # remove pre and post tokens
        reward_logits = reward_logits[:, 4:]
        hyps = hyps[:, 4:-1] 

        logprob = reward_logits.log_softmax(-1)
        token_ids = hyps[:,:,None]
        logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
        logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)
        log_reward = logprob[
            :, :, self.eos_index
        ]  # logP(generated[i+1]=term | prompt + generated[:i+1])
        log_reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
        non_term_mask = (hyps != self.eos_index)
        non_term_mask = torch.cat(
            (
                non_term_mask.new_ones(non_term_mask.shape[0], 1),
                non_term_mask,
            ),
            dim=-1,
        )  # Start (i.e., empty) state has never terminated
        log_reward[~non_term_mask] = 0.0
        log_reward_unpenalized = log_reward.clone()
        log_reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_decode_steps, -99, log_reward)

        return hyps, log_pf, log_probs_term, log_reward, log_reward_unpenalized


