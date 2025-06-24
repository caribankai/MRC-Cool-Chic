from pathlib import Path
import numpy as np

################################ Select Config #####################################

Trial = "Trial_4"
anchor = "Depth_0"
method = "Depth_6"

context_plot = 32
type = "kodak"
num_qps = 5

#####################################################################################

root = Path.home() / "OneDrive - TUM" / "Multi-Latent_Pred_for_INCs" / "Evaluations"
cntxt_str = f"{context_plot}_context_pxls"

# Define paths
version_paths = {
    anchor: root / anchor,
    method: root / Trial / method,
}

loss_big = []
count = 0

for version_name, version_path in version_paths.items():
    losses = []

    context_dir = version_path / cntxt_str
    if not context_dir.exists():
        print(f"⚠️ Context dir not found: {context_dir}")
        loss_big.append([])
        continue

    for image_dir in sorted(context_dir.iterdir()):
        if not image_dir.is_dir():
            continue

        image_name = image_dir.name

        for qp_idx in range(num_qps):
            qp_name = f"qp_{qp_idx}"
            qp_dir = image_dir / qp_name
            if not qp_dir.is_dir():
                continue

            for trial_dir in sorted(qp_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue

                count += 1  # 🔁 Increase count at each trial

                ############ Put the idx you desire here (from console output max value) and it returns the name and qp of the image ######
                idx = 115
                if count == idx + 1:
                #################################################
                    print(f"Result found at: image: {image_name}, qp: {qp_name}, trial: {trial_dir.name}")

                results_file = trial_dir / "0000-results_best.tsv"
                if results_file.exists():
                    with open(results_file, "r", encoding="utf-8") as f:
                        header = f.readline().strip().split()
                        values = f.readline().strip().split()
                        row = dict(zip(header, values))

                        if "loss" in row:
                            losses.append(float(row["loss"]))
                else:
                    print(f"⚠️ Missing: {results_file}")

    loss_big.append(losses)

#########################################
# Process Loss Ratio
#########################################

if len(loss_big) != 2:
    raise ValueError("Expected exactly two loss sets")

loss_anchor = np.array(loss_big[0])
loss_method = np.array(loss_big[1])

if len(loss_anchor) != len(loss_method):
    raise ValueError(f"Mismatch in loss lengths: {len(loss_anchor)} vs {len(loss_method)}")

loss_ratio = loss_anchor / loss_method
sorted_indices = np.argsort(loss_ratio)[::-1]  # Descending

# Top 2
top_idx = sorted_indices[0]

print(f"📍 Max loss ratio (method / anchor): {loss_ratio[top_idx]:.4f} at index {top_idx}")