import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict
import math
import torch.nn.init as init
from tqdm.auto import tqdm # Import tqdm for progress bar
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import seaborn as sns

valid_layers = [i for i in range(12) if i % 4 == 0]
layer_to_cache_idx = {layer_id: idx for idx, layer_id in enumerate(valid_layers)}

cka_cache = [[[] for _ in range(12)] for _ in valid_layers]
origin_cache = [[[] for _ in range(12)] for _ in valid_layers]

cka_data = {}

def get_model(type='gpt2'):
    if type == 'gpt2':
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        return model, tokenizer
    
def make_attn_hook(layer_id, num_heads):
    global origin_cache

    def hook(module, input, output):
        x = input[0]  # [B, T, D]
        B, T, D = x.shape
        head_dim = D // num_heads

        # Get Q, K, V
        qkv = module.c_attn(x)
        q, k, v = qkv.split(D, dim=2)  # each [B, T, D]

        def split(t):  # [B, T, D] → [B, H, T, d_head]
            return t.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3)

        Q, K, V = map(split, (q, k, v))
        attn_weights = (Q @ K.transpose(-1, -2)) / (head_dim ** 0.5)
        attn_probs = attn_weights.softmax(dim=-1)
        head_outs = attn_probs @ V  # [B, H, T, d_head]

        for h in range(num_heads):
            head_out = head_outs[:, h, :, :].reshape(-1, head_dim)
            idx = layer_to_cache_idx[layer_id]
            origin_cache[idx][h].append(head_out.detach().cpu())
    return hook


    
def extract_cka(model, tokenizer):
    handles = []
    for layer_id, layer in enumerate(model.transformer.h):
        if layer_id % 4 != 0:
            continue
        attn_module = layer.attn
        hook = make_attn_hook(layer_id, num_heads=model.config.num_attention_heads)
        h = attn_module.register_forward_hook(hook)
        handles.append(h)
    
    record_initial_cka(model, tokenizer)
    
    for h in handles:
        h.remove()


def record_initial_cka(model, tokenizer):
    model.eval()
    batch_size = 8
    
    def tokenize_function(examples):
        # Ensure padding=False here, as DataCollatorForLanguageModeling handles padding to max length within batch
        return tokenizer(examples["text"], truncation=True, padding=False)
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    wiki_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[20001:21000]")

    tokenized_val = wiki_val.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = tokenized_val.filter(lambda x: len(x["input_ids"]) > 0)

    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=collator,drop_last=True)
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device) 
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, labels=labels)
            val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")


class GQAAttention(nn.Module):
    def __init__(self, orig_attn, K=4,layer_idx=0, CKA_data=False, mean_pool=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.CKA_data= CKA_data
        self.K = K
        self.embed_dim = orig_attn.embed_dim
        self.num_heads = orig_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0

        self.group_size = self.num_heads // K

        c_attn_weight = orig_attn.c_attn.weight.data
        c_attn_bias = orig_attn.c_attn.bias.data

        W_q = c_attn_weight[:, :self.embed_dim]
        W_k = c_attn_weight[:, self.embed_dim:2*self.embed_dim]
        W_v = c_attn_weight[:, 2*self.embed_dim:]

        b_q = c_attn_bias[:self.embed_dim]
        b_k = c_attn_bias[self.embed_dim:2*self.embed_dim]
        b_v = c_attn_bias[2*self.embed_dim:]


        W_q = W_q.contiguous().view(self.num_heads, self.head_dim, self.embed_dim)
        W_k = W_k.contiguous().view(self.num_heads, self.head_dim, self.embed_dim)
        W_v = W_v.contiguous().view(self.num_heads, self.head_dim, self.embed_dim)

        # Reshape biases into [num_heads, head_dim]
        b_q = b_q.contiguous().view(self.num_heads, self.head_dim)
        b_k = b_k.contiguous().view(self.num_heads, self.head_dim)
        b_v = b_v.contiguous().view(self.num_heads, self.head_dim)

        # Mean-pool by contiguous groups
        def group_avg(W):
            return torch.stack([
                W[i*self.group_size:(i+1)*self.group_size].mean(0) for i in range(K)
            ])

        if mean_pool:
            self.W_q = nn.Parameter(W_q)
            self.W_k = nn.Parameter(group_avg(W_k))
            self.W_v = nn.Parameter(group_avg(W_v))

            self.b_q = nn.Parameter(b_q)
            self.b_k = nn.Parameter(group_avg(b_k))
            self.b_v = nn.Parameter(group_avg(b_v))
            
        else:
            # Init the weights randomly
            self.W_q = nn.Parameter(init.xavier_normal_(torch.empty_like(W_q)))
            self.W_k = nn.Parameter(init.xavier_normal_(torch.empty_like(group_avg(W_k))))
            self.W_v = nn.Parameter(init.xavier_normal_(torch.empty_like(group_avg(W_v))))

            self.b_q = nn.Parameter(init.xavier_normal_(torch.empty_like(b_q)))
            self.b_k = nn.Parameter(init.xavier_normal_(torch.empty_like(group_avg(b_k))))
            self.b_v = nn.Parameter(init.xavier_normal_(torch.empty_like(group_avg(b_v))))


        self.out_proj = orig_attn.c_proj # reuse output projection

    def forward(self, x, attention_mask=None, **kwargs):
      global cka_cache
      B, T, C = x.size()
      device = x.device

      q = torch.stack([
          F.linear(x, self.W_q[i], self.b_q[i])  # W_q[i]: [head_dim, embed_dim]
          for i in range(self.num_heads)
      ], dim=1)  # [B, num_heads, T, head_dim]

      k = torch.stack([
          F.linear(x, self.W_k[i], self.b_k[i])  # W_k[i]: [head_dim, embed_dim]
          for i in range(self.K)
      ], dim=1)  # [B, K, T, head_dim]

      v = torch.stack([
          F.linear(x, self.W_v[i], self.b_v[i])  # W_v[i]: [head_dim, embed_dim]
          for i in range(self.K)
      ], dim=1)  # [B, K, T, head_dim]

      group_indices = torch.arange(self.num_heads, device=device) // self.K
      k = k[:, group_indices, :, :]  # [B, num_heads, T, head_dim]
      v = v[:, group_indices, :, :]  # [B, num_heads, T, head_dim]

      scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, T, T]
      if attention_mask is not None:
          scores += attention_mask
      attn = torch.softmax(scores, dim=-1)  # [B, num_heads, T, T]

      out = torch.matmul(attn, v)  # [B, num_heads, T, head_dim]
      
      if not self.training and self.CKA_data:
          for h in range(out.shape[1]):  # loop over heads
              head_out = out[:, h, :, :]          # shape: [B, T, d_head]
              head_out_flat = head_out.reshape(-1, head_out.shape[-1])  # [B×T, d_head]
            
            # Save to cka_cache
              if self.layer_idx in layer_to_cache_idx:
                idx = layer_to_cache_idx[self.layer_idx]
                cka_cache[idx][h].append(head_out_flat.detach().cpu())
            
      out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, embed_dim]

      return self.out_proj(out), attn
  
  
def train_model(model,tokenizer, mat_evolution, CKA_mode=False):
    global cka_cache
    global cka_data

    batch_size = 2
    
    def tokenize_function(examples):
        # Ensure padding=False here, as DataCollatorForLanguageModeling handles padding to max length within batch
        return tokenizer(examples["text"], truncation=True, padding=False)
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    val_subset = 21000 if CKA_mode else 24000

    wiki_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20001]")
    wiki_val = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[20001:{val_subset}]")

    tokenized_train = wiki_train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_train = tokenized_train.filter(lambda x: len(x["input_ids"]) > 0)

    tokenized_val = wiki_val.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = tokenized_val.filter(lambda x: len(x["input_ids"]) > 0)

    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collator, drop_last=True)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=collator,drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)    
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device) # Move the model to the chosen device

    # number of epochs for training
    num_epochs = 5

    print(f"Starting training on {device} for {num_epochs} epochs...")

    # Store initial weights before training starts
    for i in range(len(model.transformer.h)):
        mat_evolution[i] = {
            'W_q': [],
            'W_k': [],
            'W_v': [],
        }

    W_q, W_k, W_v = model.transformer.h[i].attn.W_q, model.transformer.h[i].attn.W_k, model.transformer.h[i].attn.W_v
    mat_evolution[i]['W_q'].append(W_q.detach().cpu().numpy())
    mat_evolution[i]['W_k'].append(W_k.detach().cpu().numpy())
    mat_evolution[i]['W_v'].append(W_v.detach().cpu().numpy())


    log = {'train': [], 'val':[]}
    
    for epoch in range(num_epochs):
        model.train()

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            # Move batch tensors to the appropriate device

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, labels=labels)

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (step + 1))

            if step == len(train_loader) -1:
                for i in range(len(model.transformer.h)):
                    W_q, W_k, W_v = model.transformer.h[i].attn.W_q, model.transformer.h[i].attn.W_k, model.transformer.h[i].attn.W_v
                    mat_evolution[i]['W_q'].append(W_q.detach().cpu().numpy())
                    mat_evolution[i]['W_k'].append(W_k.detach().cpu().numpy())
                    mat_evolution[i]['W_v'].append(W_v.detach().cpu().numpy())

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")
        log["train"].append(avg_epoch_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        cka_data[epoch] = cka_cache
        cka_cache = [[[] for _ in range(12)] for _ in valid_layers]
        print(f"Validation Loss: {avg_val_loss:.4f}")
        log["val"].append(avg_val_loss)


    model.eval()
    print("\nTraining complete!")
    return log


# Compute linear CKA
def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    dot = (Y.T @ X).norm("fro")**2
    norm_x = (X.T @ X).norm("fro")
    norm_y = (Y.T @ Y).norm("fro")
    return dot / (norm_x * norm_y)


def plot_cka_heatmaps(heads1, heads2, titlex='', titley='', title=''):
    num_valid_layers = len(heads1)
    num_heads = len(heads1[0])

    for layer_idx in range(num_valid_layers):
        cka_matrix = torch.full((num_heads, num_heads), float("nan"))

        for i in range(num_heads):
            X_list = heads1[layer_idx][i]
            if not X_list:
                continue
            X = torch.cat(X_list, dim=0)

            for j in range(num_heads):
                Y_list = heads2[layer_idx][j]
                if not Y_list:
                    continue
                Y = torch.cat(Y_list, dim=0)

                # Compute linear CKA similarity
                cka_matrix[j, i] = linear_cka(X, Y)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cka_matrix.numpy(),
            xticklabels=[f"H{i}" for i in range(num_heads)],
            yticklabels=[f"H{j}" for j in range(num_heads)],
            cmap="viridis", annot=True, fmt=".2f",
            mask=torch.isnan(cka_matrix).numpy()
        )
        plt.gca().invert_yaxis()  

        actual_layer = valid_layers[layer_idx]
        plt.title(f"Layer {actual_layer} - " + title)
        plt.xlabel(titlex)
        plt.ylabel(titley)
        plt.tight_layout()
        plt.show()
