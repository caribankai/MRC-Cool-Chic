import subprocess
import time
from pathlib import Path

###########
# Still finish original 32 context pixels!

def build_all_commands():
    lambda_values = [0.02, 0.004, 0.001, 0.0004, 0.0001]
    arm_values = [24, 8, 16, 32]
    input_dirs = ["archive"]

    commands = []
    for arm in arm_values:
        for qp_idx, lmbda in enumerate(lambda_values):
            for input_dir in input_dirs:
                for image_path in Path(f"/home/cari_wiedemann/Cool-Chic/images/{input_dir}").glob("*.png"):
                    image_name = image_path.stem
                    workdir = f"/home/cari_wiedemann/Cool-Chic/Evaluations/Ours/{arm}_context_pxls/{image_name}/qp_{qp_idx}/trial_0"
                    Path(workdir).mkdir(parents=True, exist_ok=True)

                    cmd = (
                        f"python coolchic/encode.py "
                        f"-i='{image_path}' "
                        f"-o='/home/cari_wiedemann/Cool-Chic/utilities/output_bitstreams/bitstream.cool' "
                        f"--workdir='{workdir}' "
                        f"--enc_cfg=cfg/enc/intra/medium_30k.cfg "
                        f"--dec_cfg_residue=cfg/dec/intra_residue/hop.cfg "
                        f"--arm_residue={arm},2 "
                        f"--pred_depth=1 "
                        f"--pred_forward=0 "
                        f"--lmbda={lmbda}"
                    )

                    commands.append(cmd)
    return commands


def get_active_screens():
    result = subprocess.run(['screen', '-ls'], capture_output=True, text=True)
    lines = result.stdout.splitlines()
    return [line for line in lines if '\t' in line and 'codec_job_' in line]


def run_orchestrator():
    gpu_ids = [0, 1]
    commands = build_all_commands()
    max_parallel = 2
    job_counter = 0
    total_jobs = len(commands)

    while commands or get_active_screens():
        active_screens = get_active_screens()
        available_slots = max_parallel - len(active_screens)

        for _ in range(min(available_slots, len(commands))):
            cmd = commands.pop(0)
            gpu_id = gpu_ids[job_counter % len(gpu_ids)]  # Alternate between GPUs

            # Prefix the command with the desired GPU assignment
            full_cmd = (
                f"screen -dmS codec_job_{job_counter} bash -c "
                f"'CUDA_VISIBLE_DEVICES={gpu_id} {cmd}; echo DONE'"
            )

            subprocess.run(full_cmd, shell=True)
            job_counter += 1
            print(f"\r🟢 Launched jobs: {job_counter}/{total_jobs}\033[K", end='', flush=True)

        time.sleep(3)  # Give time for screen jobs to start and finish

    print("✅ All jobs finished.")


if __name__ == "__main__":
    run_orchestrator()


###### Usage
# Start master screen that orchestrates the sub screen training jobs
# screen -S master_screen
# conda activate coolchic
# python training_screen_orchestrator.py

#### Access (attach) a screen
#screen -r master_screen

#### Detach a screen 
# Ctrl + A & D

#### Kill the screens
# screen -ls  # make sure to kill no other screens
# screen -ls | grep '\.' | awk '{print $1}' | xargs -I {} screen -S {} -X quit
