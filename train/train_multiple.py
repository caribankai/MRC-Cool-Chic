import subprocess
from pathlib import Path


def run_codec(image_path, context, lmbda, qp):
    image_name = image_path.stem
    workdir = f"/home/cari_wiedemann/Cool-Chic/Evaluations/Original/{context}_context_pxls/{image_name}/qp_{qp}/trial_0"
    Path(workdir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "coolchic/encode.py",
        f"-i={str(image_path)}",
        f"-o=/home/cari_wiedemann/Cool-Chic/utilities/output_bitstreams/bitstream.cool",
        f"--workdir={workdir}",
        "--enc_cfg=cfg/enc/intra/medium_30k.cfg",
        "--dec_cfg_residue=cfg/dec/intra_residue/hop.cfg",
        f"--arm_residue={context},2",
        "--pred_depth=0",
        "--pred_forward=0",
        f"--lmbda={lmbda}",
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Success: {image_name}, context={context}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {image_name}, context={context} - {e}")

def run_all_experiments():
    lambda_values = [0.02, 0.004, 0.001, 0.0004, 0.0001]
    arm_values = [8, 16, 24, 32]
    input_dirs = ["archive"]

    for arm in arm_values:
        for qp_idx, lmbda in enumerate(lambda_values):
            for input_dir in input_dirs:
                for image_path in Path("/home/cari_wiedemann/Cool-Chic/images/" + input_dir).glob("*.png"):
                    run_codec(image_path, arm, lmbda, qp_idx)


if __name__ == "__main__":
    run_all_experiments()
