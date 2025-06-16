import subprocess
import os


current_path = os.getcwd()

def run_codec():
    cmd = [
        "python", "coolchic/encode.py",
        f"-i={current_path}/images/kodak_crop/kodim19.png",
        f"-o={current_path}/train/train_results/output_bitstreams/bitstream.cool",
        f"--workdir={current_path}/train/train_results/trial_0/trial_0_0",
        "--enc_cfg=cfg/enc/intra/fast_10k.cfg",
        "--dec_cfg_residue=cfg/dec/intra_residue/hop.cfg",
        "--arm_residue=8,2",   
        "--pred_depth=0",
        "--pred_forward=0",
        "--lmbda=0.001",
        #"--start_lr=1e-2",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("Codec executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Codec execution failed:", e)

if __name__ == "__main__":
    run_codec()

# "--start_lr=2.5e-3",