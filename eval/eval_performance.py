from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools
import json

################################ Select Config #####################################

Trial = "Trial_4"

anchor = "Depth_0"
method = "Depth_4"

context_plot = 32
type = "kodak"
avg_all_context_sizes = False  # Only available if all context sizes available

#####################################################################################

output_dir = Path("eval/performances")
output_dir.mkdir(parents=True, exist_ok=True)

filename_anchor = f"rd_{Trial}_{anchor}_{context_plot}_{type}.json"
filename_method = f"rd_{Trial}_{method}_{context_plot}_{type}.json"
file_path_anchor = output_dir / filename_anchor
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

class Curve:
    def __init__(self, bpp, psnr, label, benchmark=False):
        self.bpp = bpp
        self.psnr = psnr
        self.label = label
        self.color = None
        self.marker = None
        self.benchmark = benchmark

def plot_curves(curves, xlabel="bpp", ylabel="PSNR in dB", title="Plot"):
    plt.figure(figsize=(8, 6))

    custom_colors = [
        "#d62728", "#e377c2", "#ff7f0e", "#1f77b4", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

    marker_cycle = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']
    colors = itertools.cycle(custom_colors)
    markers = itertools.cycle(marker_cycle)

    for curve in curves:
        curve.color = next(colors)
        curve.marker = next(markers)
        plt.plot(
            curve.bpp,
            curve.psnr,
            label=curve.label,
            color=curve.color,
            marker=None if curve.benchmark else curve.marker,
            linewidth=2,
            alpha=0.7
        )

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#################################### Anchor and Method results ##############################

# Import anchor + method data from JSON
with open(file_path_anchor, "r", encoding="utf-8") as f:
    data = json.load(f)
bpp_anchor = data["bpp"]
psnr_anchor = data["psnr"]
curve_anchor = Curve(bpp_anchor, psnr_anchor, label="Cool-Chic")

with open(file_path_method, "r", encoding="utf-8") as f:
    data = json.load(f)
bpp_method = data["bpp"]
psnr_method = data["psnr"]
curve_method = Curve(bpp_method, psnr_method, label="Cool-Chic (Ours)")

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

curves = [curve_vtm2310, curve_mlicpp, curve_anchor, curve_method]

import bjontegaard as bd

bd_rate = bd.bd_rate(bpp_anchor, psnr_anchor, bpp_method, psnr_method, method='akima')
bd_psnr = bd.bd_psnr(bpp_anchor, psnr_anchor, bpp_method, psnr_method, method='akima')

print(f"BD-Rate: {bd_rate:.4f} %")
print(f"BD-PSNR: {bd_psnr:.4f} dB")

title_bd = f"{method} vs. Cool-Chic BD-Rate: {bd_rate:.2f} %, BD-PSNR: {bd_psnr:.2f} dB"
title_plot = str(context_plot) + " context pxls, " + type
title = title_plot + "\n" + title_bd

plot_curves(curves, title=title)
