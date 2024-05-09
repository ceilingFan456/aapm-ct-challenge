import torch
import os

# Assuming you have defined the 'device' variable somewhere in your code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = os.path.join(config.RESULTS_PATH, "operator_radon_fwd_train_phase_0", "model_weights_final.pt")
model = torch.load(model_path, map_location=device)

# Print all parameters of the model
for name, param in model.items():
    print(name, param)