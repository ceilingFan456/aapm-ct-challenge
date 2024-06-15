import os
import shutil

import matplotlib as mpl
import torch
import matplotlib.pyplot as plt

from data_management import load_ct_data
from networks import DCLsqFPB, GroupUNet, IterativeNet, RadonNet

import config

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)

mseloss = torch.nn.MSELoss(reduction="sum")

def loss_func(pred, tar):
    return mseloss(pred, tar) / pred.shape[0]

# ----- network configuration -----
subnet_params = {
    "in_channels": 6,
    "drop_factor": 0.0,
    "base_features": 32,
    "out_channels": 6,
    "num_groups": 32,
}
subnet = GroupUNet

# define operator
d = torch.load(
    os.path.join(
        config.RESULTS_PATH,
        "operator_radon_bwd_train_phase_1",
        "model_weights.pt",
    ),
    map_location=device,
)
radon_net = RadonNet.new_from_state_dict(d)
radon_net.OpR.flat = True
radon_net.OpR.filter_type = "hamming"
radon_net.freeze()
operator = radon_net.OpR.to(device)

dc_operator = DCLsqFPB(operator)
dc_operator.freeze()

it_net_params = {
    "num_iter": 4,
    "lam": 4 * [0.0],
    "lam_learnable": True,
    "final_dc": True,
    "resnet_factor": 1.0,
    "inverter": operator.inv,
    "dc_operator": dc_operator,
    "use_memory": 5,
}


# ------ construct network and train -----
subnet_tmp = subnet(**subnet_params).to(device)
it_net = IterativeNet(
    it_net_params["num_iter"] * [subnet_tmp], **it_net_params
).to(device)
it_net.load_state_dict(
    torch.load(
        os.path.join(
            config.RESULTS_PATH,
            "ItNet_mem_id{}_train_phase_1".format(0),
            "model_weights_epoch54.pt",
        ),
        map_location=torch.device(device),
    )
)


## construct U-Net
u_net_params = {
    "num_iter": 1,
    "lam": 0.0,
    "lam_learnable": False,
    "final_dc": False,
    "resnet_factor": 1.0,
    "inverter": operator.inv,
    "use_memory": 5,
}
subnet = subnet(**subnet_params).to(device)
u_net = IterativeNet(subnet, **u_net_params).to(device)
u_net.load_state_dict(
    torch.load(
        os.path.join(
            config.RESULTS_PATH,
            "UNet_mem_id{}_train_phase_1".format(0),
            "model_weights_epoch350.pt",
        ),
        map_location=torch.device(device),
    )
)


# Load validation data
val_data_params = {
    "folds": 400,
    "num_fold": 0,
    "leave_out": False,
}
val_data = load_ct_data("val", **val_data_params)
data_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)



# Create directory for saving results
results_dir = "pipeline_results"
os.makedirs(results_dir, exist_ok=True)

# Perform inference
with torch.no_grad():
    for i, (phantom, sinogram, fbp) in enumerate(data_loader):
        phantom, sinogram = phantom.to(device), sinogram.to(device)

        print("phantom shape:", phantom.shape)
        print("sinogram shape:", sinogram.shape)

        ## F result 
        radon_net.mode = "fwd"
        f_result = radon_net(phantom)
        f_result = f_result[0, 0].detach().cpu()
        print("f_result shape:", f_result.shape)

        ## FBP result
        radon_net.mode = "bwd"
        fbp_result = radon_net(sinogram)
        fbp_result = fbp_result[0, 0].detach().cpu()
        print("fbp_result shape:", fbp_result.shape)

        ## UNet result
        unet_result = u_net((phantom, sinogram))
        unet_result = unet_result[0, 0].detach().cpu()
        print("unet_result shape:", unet_result.shape)

        ## ItNet result
        itnet_result = it_net((phantom, sinogram))
        itnet_result = itnet_result[0, 0].detach().cpu()
        print("itnet_result shape:", itnet_result.shape)

        # Visualize and save combined image
        fig, axes = plt.subplots(2, 5, figsize=(60, 15))
        v_min = -10000000000
        v_max = 10000000000

        diff = f_result - sinogram[0, 0].detach().cpu()
        v_min = max(v_min, diff.min().item())
        v_max = min(v_max, diff.max().item())
        diff = unet_result - phantom[0, 0].detach().cpu()
        v_min = max(v_min, diff.min().item())
        v_max = min(v_max, diff.max().item())
        diff = itnet_result - phantom[0, 0].detach().cpu()
        v_min = max(v_min, diff.min().item())
        v_max = min(v_max, diff.max().item())

        vv = max(abs(v_min), abs(v_max))
        v_min = -vv
        v_max = vv

        ## first column is sinogram and ground truth 
        ax = axes[0, 0]
        im = ax.imshow(sinogram[0, 0].detach().cpu(), cmap='viridis')
        ax.set_title("sino")
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')

        ax = axes[1, 0]
        im = ax.imshow(phantom[0, 0].detach().cpu(), cmap='viridis')
        ax.set_title("gt")
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')
        
        ## second column is F(gt) and diff
        ax = axes[0, 1]
        im = ax.imshow(f_result, cmap='viridis')
        ax.set_title("F(gt)")
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')

        print(f_result.device)
        print(sinogram.device)

        diff = f_result - sinogram[0, 0].detach().cpu()
        ax = axes[1, 1]
        im = ax.imshow(diff, cmap='viridis')
        loss = loss_func(f_result.unsqueeze(0).unsqueeze(0), sinogram)
        ax.set_title("diff, loss = {:.2f}".format(loss.item()))
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')
        


        ## third column is  FBP(sino) and diff 
        ax = axes[0, 2]
        im = ax.imshow(fbp_result, cmap='viridis')
        ax.set_title("FBP(sino)")
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')

        diff = fbp_result - phantom[0, 0].detach().cpu()
        ax = axes[1, 2]
        im = ax.imshow(diff, cmap='viridis', vmin=v_min, vmax=v_max)
        loss = loss_func(f_result.unsqueeze(0).unsqueeze(0), sinogram)
        ax.set_title("diff, loss = {:.2f}".format(loss.item()))
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')

        ## fourth column is UNet(FBP(sino)) and diff
        ax = axes[0, 3]
        im = ax.imshow(unet_result, cmap='viridis')
        ax.set_title("UNet(FBP(sino))")
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')
    
        diff = unet_result - phantom[0, 0].detach().cpu()
        ax = axes[1, 3]
        im = ax.imshow(diff, cmap='viridis', vmin=v_min, vmax=v_max)
        loss = loss_func(f_result.unsqueeze(0).unsqueeze(0), sinogram)
        ax.set_title("diff, loss = {:.2f}".format(loss.item()))
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')

        ## fifth column is ItNet((FBP(sino))) and diff
        ax = axes[0, 4]
        im = ax.imshow(itnet_result, cmap='viridis')
        ax.set_title("ItNet(FBP(sino))")
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')

        diff = itnet_result - phantom[0, 0].detach().cpu()
        ax = axes[1, 4]
        im = ax.imshow(diff, cmap='viridis', vmin=v_min, vmax=v_max)
        loss = loss_func(f_result.unsqueeze(0).unsqueeze(0), sinogram)
        ax.set_title("diff, loss = {:.2f}".format(loss.item()))
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical')

        ## save result
        fig.savefig(os.path.join(results_dir, f'pipeline_{i}.png'))

        ## fifth column is ItNet((FBP(sino))) and diff
        print(f"Processed image {i+1}/{len(data_loader)}")

print("Inference complete. Results saved to:", results_dir)