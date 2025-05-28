import whisper
import torch
# import librosa # Not strictly needed if whisper.load_audio is sufficient
import os
import numpy as np
import h5py
import argparse
import sys
import re # Added for transcript cleaning

def ensure_dir_exists(path):
    """Ensures the directory for a given path exists."""
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

def clean_transcript(transcript_line):
    """
    Removes a trailing utterance ID pattern like (utt_id) or (anything_else)
    from a transcript line.
    """
    # Remove leading/trailing whitespace
    cleaned = transcript_line.strip()
    # Regex: \s* (zero or more whitespace)
    #        \( (literal open parenthesis)
    #        [^)]* (zero or more chars that are NOT a closing parenthesis)
    #        \) (literal closing parenthesis)
    #        $ (end of string)
    cleaned = re.sub(r'\s*\([^)]*\)$', '', cleaned)
    return cleaned.strip() # Strip again in case original was just "(id)"

def process_audio_and_save_features(
    manifest_file,
    transcript_file, # Added
    base_data_dir,
    output_h5_path,
    dataset_name_in_h5="encoder_embeddings",
    whisper_model_name="tiny",
    device_str=None,
):
    """
    Processes audio files listed in a manifest, extracts Whisper encoder embeddings,
    their original audio lengths (in samples), corresponding transcripts,
    and saves them to an HDF5 file.

    Args:
        manifest_file (str): Path to the manifest file.
                             Each line (after the first) should be: relative_path.flac\taudio_length_samples
        transcript_file (str): Path to the transcript file (.trn).
                               Each line corresponds to an audio entry in the manifest.
        base_data_dir (str): The base directory prepended to relative paths in the manifest.
        output_h5_path (str): Path to save the output HDF5 file.
        dataset_name_in_h5 (str): Name of the dataset for embeddings within the HDF5 file.
        whisper_model_name (str): Name of the Whisper model to use.
        device_str (str, optional): Device to run Whisper on.
    """

    ensure_dir_exists(output_h5_path)

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = whisper.load_model(whisper_model_name, device=device)
        print(f"Whisper model '{whisper_model_name}' loaded on {device}")
    except Exception as e:
        print(f"Error loading Whisper model '{whisper_model_name}' on {device}: {e}")
        sys.exit(1)

    audio_entries = [] # List of tuples (relative_path, audio_length_samples)
    try:
        with open(manifest_file, 'r', encoding='utf-8') as f: # Added encoding
            lines = f.readlines()
            if not lines:
                print(f"Error: Manifest file '{manifest_file}' is empty.")
                sys.exit(1)
            # Assuming the first line of manifest might be base_data_dir, skip it for entries
            manifest_header_lines = 1 if os.path.isdir(lines[0].strip()) else 0
            if manifest_header_lines > 0:
                print(f"Skipping first line of manifest as it seems to be base directory: {lines[0].strip()}")

            for line_num, line in enumerate(lines[manifest_header_lines:], start=manifest_header_lines + 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        audio_length = int(parts[1])
                        audio_entries.append((parts[0], audio_length))
                    except ValueError:
                        print(f"Warning: Could not parse audio length on line {line_num} of manifest: '{line}'. Skipping.")
                        continue
                else:
                    print(f"Warning: Line {line_num} in manifest does not have at least two tab-separated parts: '{line}'. Skipping.")
    except FileNotFoundError:
        print(f"Error: Manifest file '{manifest_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading manifest file '{manifest_file}': {e}")
        sys.exit(1)

    if not audio_entries:
        print("No valid audio entries found in the manifest.")
        sys.exit(0)

    num_audio_files = len(audio_entries)
    print(f"Found {num_audio_files} audio entries in the manifest.")

    # --- Read Transcripts ---
    raw_transcript_lines = []
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f_trn: # Added encoding
            raw_transcript_lines = [line for line in f_trn.readlines()] # Keep all lines for now
    except FileNotFoundError:
        print(f"Error: Transcript file '{transcript_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading transcript file '{transcript_file}': {e}")
        sys.exit(1)

    if len(raw_transcript_lines) != num_audio_files:
        print(f"Error: Number of audio entries in manifest ({num_audio_files}) "
              f"does not match number of lines in transcript file ({len(raw_transcript_lines)}).")
        print("Please ensure the transcript file has one line per valid audio entry from the manifest (after skipping manifest header).")
        sys.exit(1)
    print(f"Successfully read {len(raw_transcript_lines)} transcript lines.")


    # --- Determine embedding dimensions (once) ---
    fixed_encoder_time_dim = 1500
    embedding_dim = 0
    try:
        dummy_mel = torch.randn(1, model.dims.n_mels, 3000).to(device)
        with torch.no_grad():
            dummy_features = model.embed_audio(dummy_mel)
        embedding_dim = dummy_features.shape[-1]
        fixed_encoder_time_dim = dummy_features.shape[1]
        del dummy_mel, dummy_features
    except Exception as e:
        print(f"Warning: Could not determine embedding dim dynamically: {e}. Using model.dims.n_audio_state if available.")
        if hasattr(model.dims, 'n_audio_state'):
            embedding_dim = model.dims.n_audio_state
        else:
            dims_map = {"tiny": 384, "base": 512, "small": 768, "medium": 1024, "large": 1280, "large-v2": 1280, "large-v3": 1280}
            clean_model_name = whisper_model_name.split('.')[0]
            if clean_model_name in dims_map:
                embedding_dim = dims_map[clean_model_name]
            else:
                print(f"Error: Could not determine embedding dimension for model '{whisper_model_name}'.")
                sys.exit(1)

    feature_dtype_np = np.float32
    print(f"Expected encoder embedding shape per file: ({fixed_encoder_time_dim}, {embedding_dim}), dtype: {feature_dtype_np}")

    # --- Create HDF5 file ---
    final_h5_shape_embeddings = (num_audio_files, fixed_encoder_time_dim, embedding_dim)
    h5_chunk_shape_embeddings = (1, fixed_encoder_time_dim, embedding_dim)
    dt_str_h5 = h5py.string_dtype(encoding='utf-8') # For paths and transcripts

    try:
        with h5py.File(output_h5_path, 'w') as f_h5:
            embeddings_dset = f_h5.create_dataset(
                dataset_name_in_h5,
                shape=final_h5_shape_embeddings,
                dtype=feature_dtype_np,
                chunks=h5_chunk_shape_embeddings
            )
            paths_dset = f_h5.create_dataset("relative_filepaths", shape=(num_audio_files,), dtype=dt_str_h5)
            lengths_dset = f_h5.create_dataset("wav_lens", shape=(num_audio_files,), dtype=np.int64)
            transcripts_dset = f_h5.create_dataset("transcripts", shape=(num_audio_files,), dtype=dt_str_h5) # New dataset

            # --- Process each audio file and write to HDF5 ---
            for idx, (rel_path, audio_length_samples) in enumerate(audio_entries):
                full_audio_path = os.path.join(base_data_dir, rel_path)
                current_raw_transcript = raw_transcript_lines[idx]
                processed_transcript = clean_transcript(current_raw_transcript)

                print(f"Processing ({idx+1}/{num_audio_files}): {full_audio_path} (Len: {audio_length_samples} samples)", end=" ")

                try:
                    audio_np = whisper.load_audio(full_audio_path)
                    audio_np_padded = whisper.pad_or_trim(audio_np)
                    mel_spectrogram = whisper.log_mel_spectrogram(
                        audio_np_padded,
                        n_mels=model.dims.n_mels,
                        padding=0
                    ).to(model.device)

                    with torch.no_grad():
                        features_batch = model.embed_audio(mel_spectrogram[None])
                    current_features_np = features_batch.squeeze(0).cpu().numpy()

                    if current_features_np.shape[0] != fixed_encoder_time_dim or \
                       current_features_np.shape[1] != embedding_dim:
                        print(f"\nWarning: Feature shape mismatch for {rel_path}. "
                              f"Expected ({fixed_encoder_time_dim},{embedding_dim}), "
                              f"got {current_features_np.shape}. Filling with zeros.")
                        embeddings_dset[idx, :, :] = np.zeros((fixed_encoder_time_dim, embedding_dim), dtype=feature_dtype_np)
                    else:
                        embeddings_dset[idx, :, :] = current_features_np

                    paths_dset[idx] = rel_path
                    lengths_dset[idx] = audio_length_samples
                    transcripts_dset[idx] = processed_transcript # Store processed transcript
                    print(f"-> Stored. Embeddings: {current_features_np.shape}")

                    del audio_np, audio_np_padded, mel_spectrogram, features_batch, current_features_np

                except FileNotFoundError:
                    print(f"\nError: Audio file not found: {full_audio_path}. Storing transcript, filling embeddings with zeros.")
                    embeddings_dset[idx, :, :] = np.zeros((fixed_encoder_time_dim, embedding_dim), dtype=feature_dtype_np)
                    paths_dset[idx] = rel_path # Store original rel_path
                    lengths_dset[idx] = -1 # Indicate audio error
                    transcripts_dset[idx] = processed_transcript # Still store the transcript
                except Exception as e:
                    print(f"\nError processing {full_audio_path}: {e}. Storing transcript, filling embeddings with zeros.")
                    embeddings_dset[idx, :, :] = np.zeros((fixed_encoder_time_dim, embedding_dim), dtype=feature_dtype_np)
                    paths_dset[idx] = rel_path # Store original rel_path
                    lengths_dset[idx] = -1 # Indicate audio error
                    transcripts_dset[idx] = processed_transcript # Still store the transcript

    except Exception as e:
        print(f"An error occurred during HDF5 file creation or writing: {e}")
        sys.exit(1)

    print(f"\nAll data (embeddings, paths, lengths, transcripts) successfully saved to HDF5 file: {output_h5_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract Whisper encoder embeddings, audio lengths, and transcripts; save to HDF5.")
    parser.add_argument("manifest_file", type=str, help="Path to the manifest file (audio paths and lengths).")
    parser.add_argument("transcript_file", type=str, help="Path to the transcript file (.trn).") # New argument
    parser.add_argument("output_h5_path", type=str, help="Path to save the output HDF5 file.")
    parser.add_argument(
        "--base_data_dir", type=str, default=None,
        help="Base directory for audio files. Inferred from manifest's first line if it's a directory and this is None."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="encoder_embeddings",
        help="Name of the embeddings dataset within HDF5 (default: encoder_embeddings)."
    )
    parser.add_argument(
        "--whisper_model", type=str, default="tiny",
        choices=["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en", "large.en", "large-v2", "large-v3"],
        help="Whisper model name (default: tiny)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (e.g., 'cpu', 'cuda'). Auto-detects if None."
    )

    args = parser.parse_args()

    base_data_dir_to_use = args.base_data_dir
    if not base_data_dir_to_use:
        try:
            with open(args.manifest_file, 'r', encoding='utf-8') as f: # Added encoding
                first_line = f.readline().strip()
                if os.path.isdir(first_line):
                    base_data_dir_to_use = first_line
                    print(f"Inferred base_data_dir from manifest's first line: {base_data_dir_to_use}")
                else:
                    print("Error: --base_data_dir not provided and could not infer from manifest's first line (it's not a directory path).")
                    sys.exit(1)
        except FileNotFoundError:
            print(f"Error: Manifest file '{args.manifest_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading manifest file to infer base_data_dir: {e}")
            sys.exit(1)

    process_audio_and_save_features(
        manifest_file=args.manifest_file,
        transcript_file=args.transcript_file, # Pass transcript file
        base_data_dir=base_data_dir_to_use,
        output_h5_path=args.output_h5_path,
        dataset_name_in_h5=args.dataset_name,
        whisper_model_name=args.whisper_model,
        device_str=args.device
    )

if __name__ == "__main__":
    main()
