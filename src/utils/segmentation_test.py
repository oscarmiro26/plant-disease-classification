import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 
from src.data.svm_preprocessing import segment_leaf, segment_leaf 
from src.training.config import PLOTS_DIR


def test_segmentation():
    image_path = "data/raw/color/Cherry_(including_sour)___healthy/0a0bd696-c093-47ef-866b-7f5a40af3edb___JR_HL 3952.JPG"
    try:
        pil_image = Image.open(image_path)
        rgb_image = np.array(pil_image.convert("RGB"))
        print(f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    try:
        # Experiment with gaussian_sigma. Common values are between 1 and 3.
        # Based on your image, the leaf is likely darker than the background in grayscale,
        # so invert_mask=True should make the leaf white.
        segmented_image = segment_leaf(rgb_image)
        print(f"Segmented mask (from RGB) shape: {segmented_image.shape}, dtype: {segmented_image.dtype}")
        print(f"Unique values in segmented mask (from RGB): {np.unique(segmented_image)}")

        # Display results for RGB
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_image)
        axes[0].set_title('Original RGB Image')
        axes[0].axis('off')

        axes[1].imshow(segmented_image)
        axes[1].set_title('Segmented Mask')
        axes[1].axis('off')
        
        fig.suptitle("RGB Image Segmentation Test") # Use fig.suptitle for the main title
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

        # --- Save the figure ---
        output_filename = "segmentation_output.png"
        output_path = os.path.join(PLOTS_DIR, output_filename)
        
        try:
            plt.savefig(output_path)
            print(f"Saved segmentation output to: {output_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
        # --- End save ---

        plt.show()
    except Exception as e:
        print(f"Error during RGB image segmentation or display: {e}")


if __name__ == "__main__":
    test_segmentation()