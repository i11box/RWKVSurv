import math, sys, datetime
import logging
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
logger = logging.getLogger(__name__)

# print('logging to wandb... (comment it if you don\'t have wandb)')
# import wandb # comment this if you don't have wandb

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=7.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight  # 少数类的权重，设为多数类与少数类的比例
        
    def forward(self, predictions, targets):
        # 计算加权二元交叉熵
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, 
            pos_weight=torch.tensor([self.pos_weight], device=predictions.device)
        )
        return loss

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    weight_decay = 0.01
    lr_decay = False # linear warmup followed by cosine decay
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper
    final_tokens = 260e9 # at which point do we reach lr_final
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, use_weighted_loss=False, pos_weight=7.0):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.avg_loss = -1
        self.steps = 0
        
        # 是否使用加权损失函数
        self.use_weighted_loss = use_weighted_loss
        if use_weighted_loss:
            self.loss_fn = WeightedBCELoss(pos_weight=pos_weight)
            print(f'使用加权损失函数，正样本权重: {pos_weight}')

        self.device = 'cpu'
        if torch.cuda.is_available(): # take over whatever gpus are on the system
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def get_run_name(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        cfg = raw_model.config
        run_name = str(cfg.vocab_size) + '-' + str(cfg.ctx_len) + '-' + cfg.model_type + '-' + str(cfg.n_layer) + '-' + str(cfg.n_embd)
        return run_name

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)
            
            losses = []
            
            for it, (static_features, dynamic_features, targets, durations) in pbar:
                static_features = static_features.to(self.device)
                dynamic_features = dynamic_features.to(self.device)
                targets = targets.to(self.device)
                durations = durations.to(self.device)
                
                with torch.set_grad_enabled(is_train):
                    if self.use_weighted_loss:
                        # 使用加权损失函数
                        aki_probs, _ = model(static_features, dynamic_features, targets, durations, is_training=is_train)
                        
                        # 计算加权二元交叉熏损失
                        loss = self.loss_fn(aki_probs.squeeze(-1), targets)
                    else:
                        # 使用模型原有的损失函数
                        _, loss = model(static_features, dynamic_features, targets, durations, is_training=is_train)
                    
                    loss = loss.mean()         # collapse all losses if they are scattered on multiple gpus

                if is_train: # backprop and update the parameters                    
                    model.zero_grad()
                    loss.backward()

                    # 应用梯度裁剪以提高稳定性
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    
                    # 使用固定学习率，移除复杂的学习率调度
                    lr = config.learning_rate

                    # 记录当前损失
                    now_loss = loss.item()
                    losses.append(now_loss)
                    
                    # 计算平均损失用于显示
                    avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else 0
                    
                    # 更新进度条
                    pbar.set_description(f"epoch {epoch+1} iter {it}: loss {now_loss:.4f} avg_loss {avg_loss:.4f} lr {lr:e}")
                    
                    self.steps += 1
            
            # 返回该轮的平均损失
            return sum(losses) / len(losses) if losses else 0

        # while True:
        for epoch in range(config.max_epochs):

            run_epoch('train')
            
            if (self.config.epoch_save_frequency > 0 and epoch % self.config.epoch_save_frequency == 0) or (epoch == config.max_epochs - 1):
                raw_model = self.model.module if hasattr(self.model, "module") else self.model # DataParallel wrappers keep raw model object in .module
                torch.save(raw_model, self.config.epoch_save_path + '.pt')
