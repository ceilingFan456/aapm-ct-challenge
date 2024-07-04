import os

import matplotlib as mpl
import torch
import torchvision
import shutil

from data_management import Permute, load_ct_data
from networks import DCLsqFPB, GroupUNet, RadonNet, IterativeNet

import pandas as pd



# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)


# ----- network configuration -----
radon_params = {
    "n": [512, 512],
    "n_detect": 1024,
    "angles": torch.linspace(0, 360, 129, requires_grad=False)[:-1],
    "d_source": torch.tensor(1000.00, requires_grad=False),
    "s_detect": torch.tensor(-1.0, requires_grad=False),
    "scale": torch.tensor(0.01, requires_grad=False),
    "flat": True,
    "mode": "fwd",
}
radon_net = RadonNet

# specify block coordinate descent order
def _specify_param(radon_net, train_phase):
    if train_phase % 3 == 0:
        radon_net.OpR.angles.requires_grad = False
        radon_net.OpR.d_source.requires_grad = False
        radon_net.OpR.scale.requires_grad = True
    elif train_phase % 3 == 1:
        radon_net.OpR.angles.requires_grad = False
        radon_net.OpR.d_source.requires_grad = True
        radon_net.OpR.scale.requires_grad = False
    elif train_phase % 3 == 2:
        radon_net.OpR.angles.requires_grad = True
        radon_net.OpR.d_source.requires_grad = False
        radon_net.OpR.scale.requires_grad = False

# ----- fwd training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")

def loss_func(pred, tar):
    return mseloss(pred, tar) / pred.shape[0]

fwd_train_phases = 3 * 10
fwd_train_params = {
    "num_epochs": int(fwd_train_phases / 3) * [3, 2, 1],
    "batch_size": fwd_train_phases * [20],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "operator_radon_{}_"
            "train_phase_{}".format(
                radon_params["mode"], (i + 1) % (fwd_train_phases + 1),
            ),
        )
        for i in range(fwd_train_phases + 1)
    ],
    "save_epochs": 10,
    "optimizer": torch.optim.Adam,
    "optimizer_params": int(fwd_train_phases / 3)
    * [
        {"lr": 1e-4, "eps": 1e-5},
        {"lr": 1e-0, "eps": 1e-5},
        {"lr": 1e-1, "eps": 1e-5},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 50, "gamma": 0.75},
    "acc_steps": fwd_train_phases * [1],
    "train_transform": torchvision.transforms.Compose([Permute([2, 1])]),
    "val_transform": torchvision.transforms.Compose([Permute([2, 1])]),
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
}


# ----- data configuration -----

# always use same folds, num_fold for noth train and val
# always use leave_out=True on train and leave_out=False on val data
train_data_params = {
    "folds": 2,
    "num_fold": 0,
    "leave_out": True,
}
val_data_params = {
    "folds": 2,
    "num_fold": 0,
    "leave_out": False,
}

train_data = load_ct_data("train", **train_data_params)
val_data = load_ct_data("train", **val_data_params)

# ------ save hyperparameters -------
os.makedirs(fwd_train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(fwd_train_params["save_path"][-1], "fwd_hyperparameters.txt"), "w"
) as file:
    for key, value in radon_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in fwd_train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(fwd_train_phases) + "\n")



###############################################################
## 
## Train forward operator 
## 
###############################################################


# ------ construct network and train -----
radon_net = radon_net(**radon_params).to(device)
print(list(radon_net.parameters()))

logging = pd.DataFrame(
    columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error"]
)

for i in range(fwd_train_phases):
    train_params_cur = {}
    for key, value in fwd_train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    _specify_param(radon_net, i)

    log = radon_net.train_on(train_data, val_data, **train_params_cur)
    print(f"log = {log}")
    logging = pd.concat([logging, log], ignore_index=True)    
    print(f"logging={logging}")

logging.to_csv('forward_log.csv', index=False)



###############################################################
## 
## Train inverse operator 
## 
###############################################################


# ----- network configuration -----

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
radon_net.mode = "bwd"
radon_net.OpR.flat = True
radon_net.OpR.filter_type = "hamming"
radon_net.freeze()

# learnable
radon_net.OpR.inv_scale.requires_grad = True

print(list(radon_net.parameters()))


train_phases = 1
train_params = {
    "num_epochs": [5],
    "batch_size": [20],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "operator_radon_{}_"
            "train_phase_{}".format(
                radon_net.mode, (i + 1) % (train_phases + 1),
            ),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 1e-3, "eps": 1e-5}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1],
    "train_transform": torchvision.transforms.Compose([Permute([1, 2])]),
    "val_transform": torchvision.transforms.Compose([Permute([1, 2])]),
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
}

# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    # for key, value in radon_params.items():
    #     file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")


# ------ construct network and train -----

logging = pd.DataFrame(
    columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error"]
)

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    log = radon_net.train_on(train_data, val_data, **train_params_cur)
    # logging = logging.append(log, ignore_index=True)
    logging = pd.concat([logging, log], ignore_index=True)    

logging.to_csv('backward_log.csv', index=False)




###############################################################
## 
## Train Unet 
## 
###############################################################

if "SGE_TASK_ID" in os.environ:
    job_id = int(os.environ.get("SGE_TASK_ID")) - 1
else:
    job_id = 0

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
        # "model_weights_epoch.pt",
        "model_weights.pt",
    ),
    map_location=device,
)
radon_net = RadonNet.new_from_state_dict(d)
radon_net.OpR.flat = True
radon_net.OpR.filter_type = "hamming"
radon_net.freeze()
operator = radon_net.OpR.to(device)

it_net_params = {
    "num_iter": 1,
    "lam": 0.0,
    "lam_learnable": False,
    "final_dc": False,
    "resnet_factor": 1.0,
    "inverter": operator.inv,
    "use_memory": 5,
}

train_phases = 1
train_params = {
    "num_epochs": [400],
    "batch_size": [30], ## default is 4
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "UNet_mem_id{}_"
            "train_phase_{}".format(job_id, (i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-3}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 4, "gamma": 0.99},
    "acc_steps": [1],
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
}


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
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)

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






###############################################################
## 
## Train ItNet
## 
###############################################################
dc_operator = DCLsqFPB(operator)
dc_operator.freeze()


it_net_params = {
    "num_iter": 3,
    "lam": [1.1183, 1.3568, 1.4271],
    "lam_learnable": True,
    "final_dc": True,
    "resnet_factor": 1.0,
    "inverter": operator.inv,
    "dc_operator": dc_operator,
    "use_memory": 5,
}



train_phases = 1
train_params = {
    "num_epochs": [250],
    "batch_size": [5],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "ItNet_3_mem_id{}_"
            "train_phase_{}".format(job_id, (i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 10,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 8e-5, "eps": 1e-5, "weight_decay": 1e-4}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 2, "gamma": 0.99},
    "acc_steps": [1],
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
}



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