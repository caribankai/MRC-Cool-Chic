import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import matplotlib.image as mpimg
import numpy as np

# Define base path
base_path = Path.home() / "OneDrive - TUM" / "Multi-Latent_Pred_for_INCs"

# Evaluation configs
depth_2, context_2, kodak_im_2, qp_2, trial_2 = 6, 32, "19", 1, "Trial_4"
depth_3, context_3, kodak_im_3, qp_3, trial_3 = 0, 32, "19", 1, ""

# Define image paths
path_1 = Path("/Users/cari/git/Cool-Chic/images/kodak/kodim19.png")  # Original
path_2 = base_path / f"Evaluations/{trial_2}/Depth_{depth_2}/{context_2}_context_pxls/kodim{kodak_im_2}/qp_{qp_2}/trial_0/0000-decoded-kodim{kodak_im_2}.png"
path_3 = base_path / f"Evaluations/{trial_3}/Depth_{depth_3}/{context_3}_context_pxls/kodim{kodak_im_3}/qp_{qp_3}/trial_0/0000-decoded-kodim{kodak_im_3}.png"

# Load images
img1 = mpimg.imread(path_1)
img2 = mpimg.imread(path_2)
img3 = mpimg.imread(path_3)

images = [img2, img1, img3]
titles = ['Method', 'Original', 'Anchor']

# Zoom cutout region (for all images)
red_box_x, red_box_y = 60, 470
red_box_width, red_box_height = 100, 100
red_box_x2, red_box_y2 = red_box_x + red_box_width, red_box_y + red_box_height

# Compute sum over channels of squared difference (error map)
err_map = np.sum((img3.astype(float) - img2.astype(float)) ** 2, axis=-1)
err_map_log = np.log(err_map + 1e-3)

images.append(err_map_log)
titles.append('Log Squared Error (Anchor vs Method)')

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

for ax, img, title in zip(axes, images, titles):
    if title.startswith('Log Squared Error'):
        im = ax.imshow(img, cmap='hot')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=14)
        ax.axis('off')

        # Draw red rectangle at (red_box_x, red_box_y) with size 100x100
        rect = Rectangle((red_box_x, red_box_y), red_box_width, red_box_height, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Create IEEE-style inset axes at upper left of each image
        axins = inset_axes(
            ax,
            width="60%", height="60%",
            loc="upper left",
            borderpad=1
        )
        axins.imshow(img, cmap='gray')
        axins.set_xlim(red_box_x, red_box_x2)
        axins.set_ylim(red_box_y2, red_box_y)  # y2, y1 because origin is top-left
        axins.axis('off')

        # Draw dashed black rectangle at (red_box_x, red_box_y) in the inset
        axins.add_patch(Rectangle((red_box_x, red_box_y), red_box_width, red_box_height,
                                  linewidth=1, edgecolor='black', linestyle='dashed', facecolor='none'))

plt.tight_layout()
plt.show()
