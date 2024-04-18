import torch
import os
import numpy as np
from torchvision.utils import save_image
from networks import RadonNet
from data_management import Permute, load_ct_data
import config

# Assuming 'RadonNet' and 'load_ct_data' have been defined in your scope as in the training script

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d = torch.load(
    os.path.join(
        config.RESULTS_PATH,
        "operator_radon_fwd_train_phase_0",
        "model_weights_final.pt",
    ),
    map_location=device,
)
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
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = radon_net(inputs)

        print(f"inputs.shape={inputs.shape} \ntargets.shape={targets.shape}")


        # Calculate the difference
        difference = torch.abs(targets - outputs)

        print(difference.shape)

        # Save images
        save_image(outputs, os.path.join(results_dir, f"output_{i}.png"))
        save_image(targets, os.path.join(results_dir, f"groundtruth_{i}.png"))
        save_image(difference, os.path.join(results_dir, f"difference_{i}.png"))

        print(f"Processed image {i+1}/{len(data_loader)}")

print("Inference complete. Results saved to:", results_dir)
