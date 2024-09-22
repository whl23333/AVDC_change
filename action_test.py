import torch
from vector_quantize_pytorch import ResidualVQ
import os
import random
from pathlib import Path
import numpy as np
import tqdm
from omegaconf import OmegaConf
from vqvae.vqvae import VqVae
import wandb
from datasets import SequentialDatasetv2, SequentialDatasetv2SameInterval
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import yaml

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def main(cfg):

    vqvae_model = VqVae(
        input_dim_h = cfg["vqvae"]["action_window_size"],
        input_dim_w = cfg["vqvae"]["act_dim"],
        n_latent_dims = cfg["vqvae"]["n_latent_dims"],
        vqvae_n_embed = cfg["vqvae"]["vqvae_n_embed"],
        vqvae_groups = cfg["vqvae"]["vqvae_groups"],
        eval = False,
        device = cfg["device"],
        act_scale = cfg["vqvae"]["act_scale"]
    )
    seed_everything(cfg["seed"])
    current_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_dir, cfg["models"]["action_decoder"]["params"]["dir"]) 
    # train_set = SequentialDatasetv2(
    #         sample_per_seq=cfg["sample_per_seq"], 
    #         path="/media/disk3/WHL/flowdiffusion/datasets/metaworld", 
    #         target_size=(128, 128),
    #         randomcrop=True
    #     )
    
    train_set = SequentialDatasetv2SameInterval(
            sample_per_seq=cfg["sample_per_seq"], 
            path="/media/disk3/WHL/flowdiffusion/datasets/metaworld", 
            target_size=(128, 128),
            frameskip=cfg["frameskip"],
            randomcrop=True
        )
    valid_n = cfg["valid_n"]
    valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
    valid_set = Subset(train_set, valid_inds)
    batch_size = 1
    dl = DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 4)
    dl = cycle(dl)
    valid_dl = DataLoader(valid_set, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 4)
    train_steps = cfg["train_steps"]
    data=torch.load(os.path.join(save_path, "trained_vqvae.pt"),map_location=cfg["device"])
    vqvae_model.vq_layer.load_state_dict(data["vq_embedding"])
    vqvae_model.vq_layer.eval()
    vqvae_model.encoder.load_state_dict(data["encoder"])
    vqvae_model.encoder.eval()
    vqvae_model.decoder.load_state_dict(data["decoder"])
    vqvae_model.decoder.eval()
    for i in tqdm.trange(train_steps):
        x, x_cond, goal, action = next(dl)
        action = action.to(torch.float32)
        action = action.to(cfg["device"])

        (
            encoder_loss,
            vq_loss_state,
            vq_code,
            vqvae_recon_loss,
        ) = vqvae_model.vqvae_update(action)  # N T D
        

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '../configs/config.yaml')
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    main(cfg)