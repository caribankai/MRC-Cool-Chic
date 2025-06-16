import torch; 
import torchvision.transforms as T; 
import numpy as np
from PIL import Image; import math; 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os

###################################################################################################

path_1 = '/Users/cari/git/Cool-Chic-Copy/Evaluations/Original/32_context_pxls/kodim19/qp_0/trial_0'
path_2 = '/Users/cari/git/Cool-Chic-Copy/Evaluations/Original/32_context_pxls/kodim19/qp_4/trial_0'

# path_1 = '/Users/cari/git/Cool-Chic-Copy/testruns/ind_server/17_orig_fordiam'
# path_2 = '/Users/cari/git/Cool-Chic-Copy/testruns/ind_server/trial_19_256itself_wow'

# path_1 = '/Users/cari/git/Cool-Chic-Copy/Evaluations/Original/32_context_pxls/kodim19/qp_1/trial_0'
# path_2 = '/Users/cari/git/Cool-Chic-Copy/Evaluations/3_OursCausalMulti/32_context_pxls/kodim19/qp_1/trial_0'

# image = "lighthouse"
# context_pixels = 8
# rate = "lowest"

#key = "residuedetailed_sent_latent"
#key = "residuedetailed_mu"
# key = "residuedetailed_scale"
# key = "residuedetailed_log_scale"
key = "residuedetailed_rate_bit"
#key = "residuedetailed_centered_latent"
# key = "residuehpfilters"

###################################################################################################

key_dict = {
    "residuedetailed_sent_latent": "Sent Latents",
    "residuedetailed_mu": "Mean",
    "residuedetailed_scale": "Scale",
    "residuedetailed_log_scale": "Log Scale",
    "residuedetailed_rate_bit": "Residual Rate Bit",
    "residuedetailed_centered_latent": "Centered Latents",
    "residuehpfilters": "HP Filters"}

color_scheme = "magma"

results_dict_1 = torch.load(f'{path_1}/0000-results_loop.pt', map_location=torch.device('cpu'))
results_dict_2 = torch.load(f'{path_2}/0000-results_loop.pt', map_location=torch.device('cpu'))

def extract_metrics(tsv_path):
    df = pd.read_csv(tsv_path, delim_whitespace=True)
    print("Columns found:", df.columns.tolist())
    metrics = {
        'rate_bpp': float(df['rate_bpp'].iloc[0]),
        'latent_bpp': float(df['latent_bpp'].iloc[0]),
        'psnr_db': float(df['psnr_db'].iloc[0])
    }
    return metrics


metrics_1 = extract_metrics(os.path.join(path_1, '0000-results_best.tsv'))
metrics_2 = extract_metrics(os.path.join(path_2, '0000-results_best.tsv'))

print("Upper:", metrics_1)
print("Lower:", metrics_2)


num_images = len(results_dict_1[key])

fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))

def normalize(arr):
    return 2 * (arr - arr.min())/ (arr.max() - arr.min() + 1e-3) - 1

################ Plot ######################
for i, tensor in enumerate(results_dict_1[key]):

    # mu = results_dict_1['residuedetailed_mu'][i]
    # scale = results_dict_1['residuedetailed_scale'][i]
    # tensor = (tensor - mu) / scale

    print(tensor.shape)
    img = tensor.cpu().detach().numpy()[0, 0, :, :]
    h, w = img.shape  # Get height and width
    # img = normalize(img)
    sum = img.sum()
    im = axes[0, i].matshow(img, cmap=color_scheme)
    axes[0, i].annotate(
        f"{h}x{w}",
        xy=(0.5, 1.02),  # x=center, y=just above the axis
        xycoords='axes fraction',
        ha='center', va='bottom',
        fontsize=13
    )
    axes[0, i].axis('off')
    fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

mu_list = results_dict_2['residuedetailed_mu'][::-1]
scale_list = results_dict_2['residuedetailed_scale'][::-1]

for i, tensor in enumerate(results_dict_2[key][::-1]):

    # mu = mu_list[i]
    # scale = scale_list[i]
    # tensor = (tensor - mu) / scale

    img = tensor.cpu().detach().numpy()[0, 0, :, :]
    sum = img.sum()
    # img = normalize(img)
    im = axes[1, i].matshow(img, cmap=color_scheme)
    # axes[1, i].set_title(f"(Ours)")
    axes[1, i].axis('off')
    fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

prefix1 = "Top:       Original"
prefix2 = "Bottom: Ours     "

# title = (
#     f"Latents: {key_dict[key]}\n"
#     f"{prefix1}  (PSNR: {metrics_1['psnr_db']:.2f} dB, rate bpp: {metrics_1['rate_bpp']:.4f}, latent bpp: {metrics_1['latent_bpp']:.4f})\n"
#     f"{prefix2}  (PSNR: {metrics_2['psnr_db']:.2f} dB, rate bpp: {metrics_2['rate_bpp']:.4f}, latent bpp: {metrics_2['latent_bpp']:.4f})"
# )
# fig.suptitle(title, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
plt.show()







# Nice results
# path_1 = '/Users/cari/git/Cool-Chic-Copy/testruns/ind_server/17_orig_fordiam'
# path_2 = '/Users/cari/git/Cool-Chic-Copy/testruns/ind_server/trial_19_256itself_wow'