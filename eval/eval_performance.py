from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools
import json
import csv
from collections import defaultdict

################################ Select Config #####################################

anchor = "Depth_0"
method = "Depth_1"

context_plot = 24
type = "kodak"
avg_all_context_sizes = False # Only available if all context sizes available

#####################################################################################

root = Path("eval/Evaluations")

# Number of quality points
num_qps = 5

versions = {
    f"{anchor}": {}, 
    f"{method}": {},
    # "Ours_Checker": {}, 
    # "Ours_Checker_vers2": {},
    # "Ours_Single": {}
}

for version in root.iterdir():  # Versions: Original, Ours, etc.
    if not version.is_dir() or version.name not in versions:
        continue

    contexts = {
                # "8_context_pxls": {},
                # "16_context_pxls": {}, 
                "24_context_pxls": {}, 
                #"32_context_pxls": {}
            }

    for context in version.iterdir():  # Context Pixels
        if not context.is_dir() or context.name not in contexts:
            continue
                
        image_types = {#"total": {}, 
                        "kodak": {},
                        #"screen": {}
                        }
        
        # Initialize total metrics
        np_latent_bpp = np.zeros(num_qps)
        np_rate_bpp = np.zeros(num_qps)
        np_psnr = np.zeros(num_qps)

        num_images = 0
        for image in context.iterdir():                     # Images
            if image.is_dir():
                    
                # Track number of images for averaging
                num_images += 1

                # Initialize quality point lists for each metric 
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
                        for trial in qp.iterdir():          # Trials
                            if trial.is_dir():
                                
                                # track number of trials
                                num_trials += 1

                                # Open results file and read
                                results_file = trial / "0000-results_best.tsv"
                                if results_file.exists():
                                    with open(results_file, "r", encoding="utf-8") as f:
                                        header = f.readline().strip().split()
                                        values = f.readline().strip().split()
                                        row = dict(zip(header, values))

                                        # Add up metrics for length of images
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

                        # Average one qp over trials
                        trial_latent_bpp = trial_latent_bpp / num_trials
                        trial_rate_bpp = trial_rate_bpp / num_trials
                        trial_psnr = trial_psnr / num_trials

                        # Append to qps list
                        latent_bpp_list.append(trial_latent_bpp)
                        rate_bpp_list.append(trial_rate_bpp)
                        psnr_list.append(trial_psnr)

                # Sum qp points per image over trials
                np_latent_bpp += np.array(latent_bpp_list)
                np_rate_bpp += np.array(rate_bpp_list)
                np_psnr += np.array(psnr_list)

        # Average of number of images
        np_trial_latent_bpp = np_latent_bpp / num_images
        np_trial_rate_bpp = np_rate_bpp / num_images
        np_trial_psnr = np_psnr / num_images
                        
        image_types["kodak"] = {"latent_bpp": np_trial_latent_bpp, "rate_bpp": np_trial_rate_bpp, "psnr": np_trial_psnr}

        contexts[context.name] = image_types

    versions[version.name] = contexts


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
    plt.figure(figsize=(8,6))

    # Use built-in color and marker cycles
    #color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    custom_colors = [
        "#d62728",  # brick red
        "#e377c2",
        "#ff7f0e",  # safety orange
        "#1f77b4",  # muted blue
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
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
            marker = None if curve.benchmark else curve.marker,
            linewidth=2
        )

    plt.xlabel(xlabel, fontsize=15)   # increased font size
    plt.ylabel(ylabel, fontsize=15)   # increased font size
    #plt.title(title, fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=15)            # increased legend font size
    plt.grid(True)
    plt.tight_layout()
    plt.show()

########################################################### Import VTM results #######################################################

# VTM benchmark ########################################
# with open("utilities/vtm_results.json", "r") as f:
#     data = json.load(f)

# # Extract PSNR and BPP
# psnr_vtm91 = data["results"]["psnr-rgb"][:-2]
# bpp_vtm91 = data["results"]["bpp"][:-2]

# curve_vtm91 = Curve(bpp_vtm91, psnr_vtm91, label="VTM 9.1", benchmark=True)

# MLIC++ ###################################################

with open("eval/benchmarks/mlicplusplus_mse.json", "r", encoding="utf-8") as f:
    data = json.load(f)

bpp_mlicpp = data["bpp"]
psnr_mlicpp = data["psnr"]

curve_mlicpp = Curve(bpp_mlicpp, psnr_mlicpp, label="MLIC++", benchmark=True)

# VTM NEW ########################################

with open("eval/benchmarks/vtm_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

bpp_vtm2310 = data["bpp"]
psnr_vtm2310 = data["psnr"]

curve_vtm2310 = Curve(bpp_vtm2310, psnr_vtm2310, label="VTM 23.10", benchmark=True)


############################################################################################################

cntxt_str = f"{context_plot}_context_pxls"

if avg_all_context_sizes: 

    latent_bpp_anchor = np.zeros_like(np.array(versions[anchor][cntxt_str][type]["latent_bpp"]))
    rate_bpp_anchor = np.zeros_like(np.array(versions[anchor][cntxt_str][type]["rate_bpp"]))

    latent_bpp_method = np.zeros_like(np.array(versions[method][cntxt_str][type]["latent_bpp"]))
    rate_bpp_method = np.zeros_like(np.array(versions[method][cntxt_str][type]["rate_bpp"]))

    psnr_anchor = np.zeros_like(np.array(versions[anchor][cntxt_str][type]["psnr"]))
    psnr_method = np.zeros_like(np.array(versions[method][cntxt_str][type]["psnr"]))

    for i in [8,16,24,32]:
        cntxt_str = f"{i}_context_pxls"

        latent_bpp_anchor += np.array(versions[anchor][cntxt_str][type]["latent_bpp"])
        rate_bpp_anchor += np.array(versions[anchor][cntxt_str][type]["rate_bpp"])

        latent_bpp_method += np.array(versions[method][cntxt_str][type]["latent_bpp"])
        rate_bpp_method += np.array(versions[method][cntxt_str][type]["rate_bpp"])

        psnr_anchor += np.array(versions[anchor][cntxt_str][type]["psnr"])
        psnr_method += np.array(versions[method][cntxt_str][type]["psnr"])

    latent_bpp_anchor = latent_bpp_anchor / 4
    rate_bpp_anchor = rate_bpp_anchor / 4

    latent_bpp_method = latent_bpp_method / 4
    rate_bpp_method = rate_bpp_method / 4

    psnr_anchor = psnr_anchor / 4
    psnr_anchor = psnr_anchor / 4

    curve_latent_anchor = Curve(latent_bpp_anchor, psnr_anchor, label=anchor)
    curve_rate_anchor = Curve(rate_bpp_anchor, psnr_anchor, label=anchor)

    curve_latent_method = Curve(latent_bpp_method, psnr_method, label=method)
    curve_rate_method = Curve(rate_bpp_method, psnr_method, label=method)

    title_plot = "Avg. over all context pixels sizes, " + type


else: 

    ##### Single Codecs
    latent_bpp_anchor = np.array(versions[anchor][cntxt_str][type]["latent_bpp"])
    rate_bpp_anchor = np.array(versions[anchor][cntxt_str][type]["rate_bpp"])

    latent_bpp_method = np.array(versions[method][cntxt_str][type]["latent_bpp"])
    rate_bpp_method = np.array(versions[method][cntxt_str][type]["rate_bpp"])

    psnr_anchor = np.array(versions[anchor][cntxt_str][type]["psnr"])
    psnr_method = np.array(versions[method][cntxt_str][type]["psnr"])

    curve_latent_anchor = Curve(latent_bpp_anchor, psnr_anchor, label=anchor + " latent bpp")
    curve_rate_anchor = Curve(rate_bpp_anchor, psnr_anchor, label="Cool-Chic")

    curve_latent_method = Curve(latent_bpp_method, psnr_method, label=method + " latent bpp")
    curve_rate_method = Curve(rate_bpp_method, psnr_method, label="Cool-Chic (Ours)")

    title_plot = str(context_plot) + " context pxls, " + type


########## Plotting and BD rate 

# Select curves for plotting
# curves = [curve_rate_anchor, curve_rate_method, curve_latent_anchor, curve_latent_method]
curves = [curve_vtm2310, curve_mlicpp, curve_rate_anchor, curve_rate_method]

######### BD Rate for rate bpp

import bjontegaard as bd

bd_rate = bd.bd_rate(rate_bpp_anchor, psnr_anchor, rate_bpp_method, psnr_method, method='akima')
bd_psnr = bd.bd_psnr(rate_bpp_anchor, psnr_anchor, rate_bpp_method, psnr_method, method='akima')

print(f"BD-Rate: {bd_rate:.4f} %")
print(f"BD-PSNR: {bd_psnr:.4f} dB")

title_bd = f"{method} vs. Cool-Chic BD-Rate: {bd_rate:.2f} %, BD-PSNR: {bd_psnr:.2f} dB"
title = title_plot + "\n" + title_bd

# Plot Command
plot_curves(curves, title=title)


                            

                        

