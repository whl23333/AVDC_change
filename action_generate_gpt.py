import wandb
import os
from pathlib import Path
from datasets import SequentialDatasetv2, SequentialDatasetv2_continue
from torch.utils.data import DataLoader
from test import ActionGenerateModel
from transformers import GPT2Model
from transformers import GPT2Config
import torch
from torch.optim import Adam
import torch.nn.functional as F
from img_encoder import Encoder
from torch import nn
from torch.utils.data import Subset
import tqdm
from img_encoder import Encoder
from transformers import CLIPTextModel, CLIPTokenizer
def cycle(dl):
    while True:
        for data in dl:
            yield data
def encode_batch_text(tokenizer, text_encoder, batch_text):
    batch_text_ids = tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
    batch_text_embed = text_encoder(**batch_text_ids).last_hidden_state
    # print(batch_text_embed)
    return batch_text_embed
def main(cfg):
    run = wandb.init(
        project=cfg["wandb"]["wandb_project"],
        entity=cfg["wandb"]["wandb_entity"],
        config=cfg,
    )
    run_name = run.name or "Offline"
    current_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_dir, cfg["gpt_save_path"]) 
    save_path = Path(save_path)/run_name
    save_path.mkdir(parents=True, exist_ok = True)

    train_set = SequentialDatasetv2(
            sample_per_seq=cfg["sample_per_seq"], 
            path="/media/disk3/WHL/flowdiffusion/datasets/metaworld", 
            target_size=(128, 128),
            randomcrop=True
        )
    
    config = GPT2Config(
        vocab_size = 1,
        n_embd=512,
        n_head=4
    )
    img_encoder= Encoder(
            hidden_size = 512,
            activation_function = "relu",
            ch = 3,
            robot = False
        )
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    model = ActionGenerateModel(config=config)
    valid_n = cfg["valid_n"]
    valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
    valid_set = Subset(train_set, valid_inds)
    batch_size = cfg["batch_size"]
    dl = DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 4)
    dl = cycle(dl)
    valid_dl = DataLoader(valid_set, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 4)
    train_steps = cfg["train_steps"]
    for i in tqdm.trange(train_steps):
        x, x_cond, goal, action = next(dl)
        action = action.to(torch.float32)
        action = action.to(cfg["device"])
        x_cond = x_cond.to(cfg["device"])
        img_input = img_encoder(x_cond)
        text_input = encode_batch_text(tokenizer = tokenizer, text_encoder= text_encoder, batch_text = goal)
        stacked_input=torch.cat([text_input, ])
        action = action.to(cfg["device"])
        (
            encoder_loss,
            vq_loss_state,
            vq_code,
            vqvae_recon_loss,
        ) = vqvae_model.vqvae_update(action)  # N T D
        wandb.log({"pretrain/n_different_codes": len(torch.unique(vq_code))})
        wandb.log(
            {"pretrain/n_different_combinations": len(torch.unique(vq_code, dim=0))}
        )
        wandb.log({"pretrain/encoder_loss": encoder_loss})
        wandb.log({"pretrain/vq_loss_state": vq_loss_state})
        wandb.log({"pretrain/vqvae_recon_loss": vqvae_recon_loss})
        if i % 3000 == 0:
            state_dict = vqvae_model.state_dict()
            torch.save(state_dict, os.path.join(save_path, "trained_vqvae.pt"))
