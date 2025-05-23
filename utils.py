import subprocess

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
    # Construct the command.
    # It's often safer to pass command arguments as a list to subprocess.run
    # to avoid issues with spaces in paths or shell injection if paths were user-supplied.
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
        # Run the command pipeline through the shell
        # `capture_output=True` gets stdout and stderr
        # `text=True` decodes stdout/stderr as text (universal_newlines=True in older Python)
        # `shell=True` is needed to interpret pipes `|`. Use with caution.
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
        print(f"Command: {pipeline_command}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: sclite command not found. Is it in your PATH?")
        return None
    except ValueError:
        print(f"Error: Could not convert extracted WER '{wer_str}' to float.")
        return None
