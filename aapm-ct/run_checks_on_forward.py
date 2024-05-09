import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from networks import RadonNet
from data_management import Permute, load_ct_data
import config

def visualize_results(index, outputs, sinogram, difference, results_dir):
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    images = [outputs, sinogram, difference]
    titles = ['Output', 'Ground Truth', 'Difference']
    
    # Calculate the min and max across all images for a consistent scale
    vmin = min(image.min().item() for image in images)
    vmax = max(image.max().item() for image in images)

    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img[0, 0].detach().cpu(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        # Add colorbar to each subplot
        fig.colorbar(im, ax=ax, orientation='vertical')

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f'combined_{index}.png'))
    plt.close(fig)

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(
    config.RESULTS_PATH,
    "operator_radon_fwd_train_phase_0",
    "model_weights_final.pt"
)
d = torch.load(model_path, map_location=device)
radon_net = RadonNet.new_from_state_dict(d)

radon_net.to(device)

# Load validation data
val_data_params = {
    "folds": 400,
    "num_fold": 0,
    "leave_out": False,
}
val_data = load_ct_data("val", **val_data_params)
data_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)

# Create directory for saving results
results_dir = "inference_results_fwd"
os.makedirs(results_dir, exist_ok=True)

# Perform inference
with torch.no_grad():
    for i, (phantom, sinogram, fbp) in enumerate(data_loader):
        phantom, sinogram = phantom.to(device), sinogram.to(device)
        outputs = radon_net(phantom)
        difference = torch.abs(sinogram - outputs)

        # Visualize and save combined image
        visualize_results(i, outputs, sinogram, difference, results_dir)
        print(f"Processed image {i+1}/{len(data_loader)}")

print("Inference complete. Results saved to:", results_dir)
