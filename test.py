from transformers import GPT2Model
from transformers import GPT2Config
import torch
from torch.optim import Adam
import torch.nn.functional as F
from img_encoder import Encoder
from torch import nn



class ActionGenerateModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super(ActionGenerateModel, self).__init__()
        self.gpt2 = GPT2Model(config)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, input_embeds, attention_mask = None):
        outputs = self.gpt2(inputs_embeds = input_embeds, attention_mask = attention_mask)
        next_token_embed = self.linear(outputs.last_hidden_state)
        return next_token_embed

    def generate(self, input_embeds, act_len, attention_mask = None):
        generated_embeddings = []
        for _ in range(act_len):
            outputs = self.gpt2(inputs_embeds = input_embeds, attention_mask = attention_mask)
            next_token_embed = self.linear(outputs.last_hidden_state[:, -1, :])
            input_embeds = torch.cat((input_embeds, next_token_embed.unsqueeze(1)), dim = 1)
            generated_embeddings.append(next_token_embed)
        return torch.stack(generated_embeddings, dim = 1)

n = [1,2,3,4,5]
print(n[-2])