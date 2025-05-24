import subprocess
import torch
from torch.distributions import Categorical

from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.utils.data_utils import undo_padding

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
    prompt_len,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1] - prompt_len
    )
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:]
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_text
    mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1] - prompt_len
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
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)

    @torch.no_grad()
    def forward(self, enc_states, wav_len):
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

            has_ended = has_ended | (inp_tokens == self.eos_index)
            log_probs[has_ended] = -torch.inf
            inp_tokens[has_ended] = self.eos_index

            if has_ended.all() or self._check_end_condition(memory):
                break

        log_probs = torch.stack(log_probs_lst, dim=1)

        scores, predictions = log_probs.max(dim=-1)
        mask = scores == -torch.inf
        scores[mask] = 0
        predictions[mask] = self.eos_index

        (
            top_hyps,
            top_lengths,
            top_scores,
            top_log_probs,
        ) = self._get_top_prediction(predictions, scores, log_probs)

        # Convert best hypothesis to list
        hyps = undo_padding(top_hyps[:, 0], top_lengths)

        return hyps, top_lengths, top_scores, top_log_probs

