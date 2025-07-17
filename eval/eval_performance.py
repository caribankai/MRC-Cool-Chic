from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools
import json

################################ Select Config #####################################

Trial = "Trial_4"

anchor = "Depth_0"
method = "Depth_6"

context_plot = 32
type = "kodak"
avg_all_context_sizes = False  # Only available if all context sizes available

#####################################################################################

output_dir = Path("eval/performances")
output_dir.mkdir(parents=True, exist_ok=True)

filename_anchor = f"rd_{Trial}_{anchor}_{context_plot}_{type}.json"
filename_method = f"rd_{Trial}_{method}_{context_plot}_{type}.json"
file_path_anchor = Path("eval/benchmarks") / filename_anchor
file_path_method = output_dir / filename_method

#####################################################################################

# Use OneDrive as root path
root = Path.home() / "OneDrive - TUM" / "Multi-Latent_Pred_for_INCs" / "Evaluations"

# Number of quality points
num_qps = 5

versions = {
    anchor: {}, 
    method: {},
}

# Build explicit paths for anchor and method
version_paths = {
    anchor: root / anchor,
    method: root / Trial / method,
}

for version_name, version_path in version_paths.items():

    if version_name == anchor and file_path_anchor.exists():
        print(f"✅ Skipping {anchor} evaluation, JSON already exists: {file_path_anchor}")
        continue
    if version_name == method and file_path_method.exists():
        print(f"✅ Skipping {method} evaluation, JSON already exists: {file_path_method}")
        continue
    if not version_path.exists():
        print(f"⚠️ Warning: {version_name} path does not exist: {version_path}")
        continue

    contexts = {
        f"{context_plot}_context_pxls": {}, 
    }

    for context in version_path.iterdir():  # Context Pixels
        if not context.is_dir() or context.name not in contexts:
            continue

        image_types = {
            "kodak": {},
        }

        # Initialize total metrics
        np_latent_bpp = np.zeros(num_qps)
        np_rate_bpp = np.zeros(num_qps)
        np_psnr = np.zeros(num_qps)

        num_images = 0
        for image in context.iterdir():  # Images
            if image.is_dir():
                num_images += 1

                latent_bpp_list = []
                rate_bpp_list = []
                psnr_list = []

                for qp_name in ['qp_0', 'qp_1', 'qp_2', 'qp_3', 'qp_4']:
                    qp = image / qp_name
                    if qp.is_dir():

                        trial_latent_bpp = 0
                        trial_rate_bpp = 0
                        trial_psnr = 0
                        num_trials = 0

                        for trial in qp.iterdir():  # Trials
                            if trial.is_dir():
                                num_trials += 1

                                results_file = trial / "0000-results_best.tsv"
                                if results_file.exists():
                                    with open(results_file, "r", encoding="utf-8") as f:
                                        header = f.readline().strip().split()
                                        values = f.readline().strip().split()
                                        row = dict(zip(header, values))

                                        for key in ["latent_bpp", "psnr_db", "rate_bpp"]:
                                            val = float(row[key])
                                            if key == "latent_bpp":
                                                trial_latent_bpp += val
                                            elif key == "rate_bpp":
                                                trial_rate_bpp += val
                                            elif key == "psnr_db":
                                                trial_psnr += val
                                else:
                                    print(f"⚠️ Missing: {results_file}")

                        if num_trials > 0:
                            latent_bpp_list.append(trial_latent_bpp / num_trials)
                            rate_bpp_list.append(trial_rate_bpp / num_trials)
                            psnr_list.append(trial_psnr / num_trials)

                np_latent_bpp += np.array(latent_bpp_list)
                np_rate_bpp += np.array(rate_bpp_list)
                np_psnr += np.array(psnr_list)

        if num_images > 0:
            np_trial_latent_bpp = np_latent_bpp / num_images
            np_trial_rate_bpp = np_rate_bpp / num_images
            np_trial_psnr = np_psnr / num_images

            image_types["kodak"] = {
                "latent_bpp": np_trial_latent_bpp,
                "rate_bpp": np_trial_rate_bpp,
                "psnr": np_trial_psnr
            }

        contexts[context.name] = image_types

    versions[version_name] = contexts

########################################################### Make json file ##########################################################

# Write JSONs only for the versions that were evaluated
for version_name, file_path in [(anchor, file_path_anchor), (method, file_path_method)]:
    if file_path.exists():
        print(f"✅ Skipping JSON write for {version_name}, already exists: {file_path}")
        continue

    version_data = versions.get(version_name, {})
    context_data = version_data.get(f"{context_plot}_context_pxls", {})
    image_data = context_data.get(type, {})

    if not image_data:
        print(f"⚠️ No data for {version_name} at context {context_plot} and type {type}")
        continue

    bpp = image_data.get("rate_bpp", [])
    psnr = image_data.get("psnr", [])

    json_data = {
        "psnr": list(np.round(psnr, 4)),
        "bpp": list(np.round(bpp, 4)),
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
        print(f"✅ Saved: {file_path}")

########################################################### PLOTTING Module ##########################################################

import matplotlib.pyplot as plt
import itertools
import os
import os
import json
import matplotlib.pyplot as plt
import itertools

class Curve:
    def __init__(self, bpp, psnr, label, benchmark=False):
        self.bpp = bpp
        self.psnr = psnr
        self.label = label
        self.color = None
        self.marker = None
        self.benchmark = benchmark

def plot_curves(curves, xlabel="Rate [bpp]", ylabel="PSNR [dB]", title="Plot"):
    plt.figure(figsize=(8.5, 5.5)) 

    # Set color order: blue, red, pink, orange
    custom_colors = ['#d62728', '#1f77b4', '#ff7f0e', '#e377c2']
    marker_cycle = ['o', 's', '^', 'D']
    colors = iter(custom_colors)
    markers = iter(marker_cycle)

    # Assign colors and markers in the desired order
    for curve in curves:
        curve.color = next(colors)
        curve.marker = next(markers)

    plt.grid(True, zorder=0, alpha=0.3)

    handles_dict = {}

    for curve in curves:
        handle, = plt.plot(
            curve.bpp,
            curve.psnr,
            label=curve.label,
            color=curve.color,
            marker=curve.marker,
            linewidth=2.5 if not curve.benchmark else 2,
            alpha=0.85 if not curve.benchmark else 1,
            markersize=7,
            zorder=2 if not curve.benchmark else 1
        )
        handles_dict[curve.label] = handle

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Specify legend order
    legend_order = ["MLIC++", "Ours", "Cool-Chic", "VTM 23.10"]
    handles = [handles_dict[label] for label in legend_order]
    plt.legend(handles, legend_order, fontsize=16)

    plt.tight_layout()
    output_path = "eval/Exported_Images/rd_plot_32_depth6_trial4.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.show()

#################################### Anchor and Method results ##############################

with open(file_path_anchor, "r", encoding="utf-8") as f:
    data = json.load(f)
bpp_anchor = data["bpp"]
psnr_anchor = data["psnr"]
curve_anchor = Curve(bpp_anchor, psnr_anchor, label="Cool-Chic")

with open(file_path_method, "r", encoding="utf-8") as f:
    data = json.load(f)
bpp_method = data["bpp"]
psnr_method = data["psnr"]
curve_method = Curve(bpp_method, psnr_method, label="Ours")

###################################### Benchmark results ##############################

with open("eval/benchmarks/mlicplusplus_mse.json", "r", encoding="utf-8") as f:
    data = json.load(f)
bpp_mlicpp = data["bpp"]
psnr_mlicpp = data["psnr"]
curve_mlicpp = Curve(bpp_mlicpp, psnr_mlicpp, label="MLIC++", benchmark=True)

with open("eval/benchmarks/vtm_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
bpp_vtm2310 = data["bpp"]
psnr_vtm2310 = data["psnr"]
curve_vtm2310 = Curve(bpp_vtm2310, psnr_vtm2310, label="VTM 23.10", benchmark=True)

############################################################################################################

# Order: MLIC++ (blue), Ours (red), Cool-Chic (pink), VTM 23.10 (orange)
curves = [curve_mlicpp, curve_anchor, curve_method, curve_vtm2310]

import bjontegaard as bd

bd_rate = bd.bd_rate(bpp_anchor, psnr_anchor, bpp_method, psnr_method, method='akima')
bd_psnr = bd.bd_psnr(bpp_anchor, psnr_anchor, bpp_method, psnr_method, method='akima')

print(f"BD-Rate: {bd_rate:.4f} %")
print(f"BD-PSNR: {bd_psnr:.4f} dB")

title_bd = f"{method} vs. Cool-Chic BD-Rate: {bd_rate:.2f} %, BD-PSNR: {bd_psnr:.2f} dB"
title_plot = str(context_plot) + " context pxls, " + type
title = title_plot + "\n" + title_bd


plot_curves(curves)
