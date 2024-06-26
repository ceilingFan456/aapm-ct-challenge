import os
import shutil

import matplotlib as mpl
import torch

from data_management import load_ct_data
from networks import DCLsqFPB, GroupUNet, IterativeNet, RadonNet


# ----- load configuration -----
import config  # isort:skip

if "SGE_TASK_ID" in os.environ:
    job_id = int(os.environ.get("SGE_TASK_ID")) - 1
else:
    job_id = 0

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)

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
    "lam": [1.1183, 1.3568, 1.4271, 0.0808],
    "lam_learnable": True,
    "final_dc": True,
    "resnet_factor": 1.0,
    "inverter": operator.inv,
    "dc_operator": dc_operator,
    "use_memory": 5,
}

# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    return mseloss(pred, tar) / pred.shape[0]


train_phases = 1
train_params = {
    "num_epochs": [250],
    "batch_size": [5],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "ItNet_mem_id{}_"
            "train_phase_{}".format(job_id, (i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 8e-5, "eps": 1e-5, "weight_decay": 1e-4}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 2, "gamma": 0.99},
    "acc_steps": [1],
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
}

# ----- data configuration -----

# always use same folds, num_fold for noth train and val
# always use leave_out=True on train and leave_out=False on val data
train_data_params = {
    "folds": 32,
    "num_fold": job_id,
    "leave_out": True,
}
val_data_params = {
    "folds": 32,
    "num_fold": job_id,
    "leave_out": False,
}
train_data = load_ct_data("train", **train_data_params)
val_data = load_ct_data("train", **val_data_params)

# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in subnet_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in it_net_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
subnet_tmp = subnet(**subnet_params).to(device)

it_net_tmp = IterativeNet(
    subnet_tmp,
    **{
        "num_iter": 1,
        "lam": 0.0,
        "lam_learnable": False,
        "final_dc": False,
        "resnet_factor": 1.0,
        "use_memory": 5,
    }
).to(device)
it_net_tmp.load_state_dict(
    torch.load(
        os.path.join(
            config.RESULTS_PATH,
            "UNet_mem_id{}_train_phase_1".format(job_id),
            "model_weights_final.pt",
        ),
        map_location=torch.device(device),
    )
)

it_net = IterativeNet(
    it_net_params["num_iter"] * [it_net_tmp.subnet], **it_net_params
).to(device)

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    logging = it_net.train_on(train_data, val_data, **train_params_cur)


# ----- pick best weights and save them ----
epoch = logging["val_chall_err"].argmin() + 1

shutil.copyfile(
    os.path.join(
        train_params["save_path"][-2], "model_weights_epoch{}.pt".format(epoch)
    ),
    os.path.join(train_params["save_path"][-2], "model_weights_final.pt"),
)
shutil.copyfile(
    os.path.join(
        train_params["save_path"][-2], "plot_epoch{}.png".format(epoch)
    ),
    os.path.join(
        train_params["save_path"][-2], "plot_epoch_final{}.png".format(epoch)
    ),
)
