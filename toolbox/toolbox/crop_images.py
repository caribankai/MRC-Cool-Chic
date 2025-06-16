from PIL import Image
import os

# === Set input folder path here ===
input_folder = "/Users/cari/git/Cool-Chic/images/kodak"
index_zero_pxls = [120, 120]
cropped_size = [256, 256]
# ==================================

def process_image(input_path, output_path):
    with Image.open(input_path) as img:
        # Remove alpha channel if present
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Crop to 120x120 from position (240, 560)
        cropped = img.crop((index_zero_pxls[0], index_zero_pxls[1], index_zero_pxls[0] + cropped_size[0], index_zero_pxls[1] + cropped_size[1]))
        cropped.save(output_path)

def process_folder(folder):
    parent_dir = os.path.dirname(folder)
    folder_name = os.path.basename(folder)
    output_folder = os.path.join(parent_dir, f"{folder_name}_crop")
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(folder):
        if fname.lower().endswith('.png'):
            input_path = os.path.join(folder, fname)
            output_path = os.path.join(output_folder, fname)
            try:
                process_image(input_path, output_path)
                print(f"Saved cropped image: {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    process_folder(input_folder)
