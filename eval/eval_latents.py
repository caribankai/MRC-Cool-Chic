import torch; 
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

# Choose config
###################################################################################################

# Current project base path
# OneDrive base path
base_path = Path.home() / "OneDrive - TUM" / "Multi-Latent_Pred_for_INCs" 

# Evaluation 1 config (anchor)
depth_1 = 0  # 0 - 6 (0 is the original)
context_1 = 32 
kodak_im_1 = "16"  # 01 - 24
qp_1 = 0 # 0 - 4
trial_1 = ""  # Corresponds to original

# Evaluation 2 config (method)
depth_2 = 6
context_2 = 32
kodak_im_2 = "16"
qp_2 = 0
trial_2 = "Trial_4"  # Corresponds to original

### Uncomment desired key
key = "residuedetailed_sent_latent"
#key = "residuedetailed_mu"
#key = "residuedetailed_scale"
#key = "residuedetailed_log_scale"
#key = "residuedetailed_rate_bit"
#key = "residuedetailed_centered_latent"
#key = "residuehpfilters"

###################################################################################################

# Construct paths
path_1 = base_path / f"Evaluations/{trial_1}/Depth_{depth_1}/{context_1}_context_pxls/kodim{kodak_im_1}/qp_{qp_1}/trial_0"
path_2 = base_path / f"Evaluations/{trial_2}/Depth_{depth_2}/{context_2}_context_pxls/kodim{kodak_im_2}/qp_{qp_2}/trial_0"

key_dict = {
    "residuedetailed_sent_latent": "Sent Latents",
    "residuedetailed_mu": "Mean",
    "residuedetailed_scale": "Scale",
    "residuedetailed_log_scale": "Log Scale",
    "residuedetailed_rate_bit": "Residual Rate Bit",
    "residuedetailed_centered_latent": "Centered Latents",
    "residuehpfilters": "HP Filters"}

color_scheme = "RdBu"

results_dict_1 = torch.load(f'{path_1}/0000-results_loop.pt', map_location=torch.device('cpu'))
results_dict_2 = torch.load(f'{path_2}/0000-results_loop.pt', map_location=torch.device('cpu'))

def extract_metrics(tsv_path):
    df = pd.read_csv(tsv_path, delim_whitespace=True)
    print("Columns found:", df.columns.tolist())
    metrics = {
        'loss': float(df['loss'].iloc[0]),
        'rate_bpp': float(df['rate_bpp'].iloc[0]),
        'latent_bpp': float(df['latent_bpp'].iloc[0]),
        'psnr_db': float(df['psnr_db'].iloc[0])
    }
    return metrics


metrics_1 = extract_metrics(os.path.join(path_1, '0000-results_best.tsv'))
metrics_2 = extract_metrics(os.path.join(path_2, '0000-results_best.tsv'))

print("Upper (anchor):", metrics_1)
print("Lower (method):", metrics_2)


num_images = len(results_dict_1[key])

fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))

def normalize(arr):
    return 2 * (arr - arr.min())/ (arr.max() - arr.min() + 1e-3) - 1

################ Plot ######################
for i, tensor in enumerate(results_dict_1[key]):
    print(tensor.shape)
    img = tensor.cpu().detach().numpy()[0, 0, :, :]
    h, w = img.shape
    im = axes[0, i].matshow(img, cmap=color_scheme)
    axes[0, i].annotate(
        f"{h}x{w}",
        xy=(0.5, 1.02),
        xycoords='axes fraction',
        ha='center', va='bottom',
        fontsize=13
    )
    axes[0, i].axis('off')
    fig.colorbar(im, ax=axes[0, i])

for i, tensor in enumerate(results_dict_2[key]):
    img = tensor.cpu().detach().numpy()[0, 0, :, :]
    im = axes[1, i].matshow(img, cmap=color_scheme)
    axes[1, i].axis('off')
    fig.colorbar(im, ax=axes[1, i])

# Automatically adjust layout
fig.tight_layout()
plt.tight_layout()
plt.show()
