import subprocess
import os
import sys

# === Parameters ===
image_path = "images/kodak_crop/kodim19.png"
arm_residue = "24,2"
pred_depth = 2
pred_forward = 0
lmbda = 0.001
trial_folder = "trial_5" 

# === Lambda dictionary for QP mapping ===
lambda_dict = {
    0.02: 0,
    0.004: 1,
    0.001: 2,
    0.0004: 3,
    0.0001: 4,
}

def run_codec():
    current_path = os.getcwd()
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    arm_main = arm_residue.split(',')[0]
    qp_index = lambda_dict[lmbda]

    # === Construct workdir path ===
    workdir_name = f"{image_name}_{arm_main}_depth{pred_depth}_forw{pred_forward}_qp{qp_index}"
    workdir = os.path.join(current_path, "train/train_results", trial_folder, workdir_name)

    if os.path.exists(workdir):
        print(f"[INFO] Workdir already exists: {workdir}")
    else:
        os.makedirs(workdir)
        print(f"[INFO] Created workdir: {workdir}")

    cmd = [
        sys.executable, "coolchic/encode.py",
        f"-i={os.path.join(current_path, image_path)}",
        f"-o={os.path.join(current_path, 'train/train_results/output_bitstreams/bitstream.cool')}",
        f"--workdir={workdir}",
        "--enc_cfg=cfg/enc/intra/fast_10k.cfg",
        "--dec_cfg_residue=cfg/dec/intra_residue/hop.cfg",
        f"--arm_residue={arm_residue}",
        f"--pred_depth={pred_depth}",
        f"--pred_forward={pred_forward}",
        f"--lmbda={lmbda}",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Codec executed successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Codec execution failed:", e)

if __name__ == "__main__":
    run_codec()
