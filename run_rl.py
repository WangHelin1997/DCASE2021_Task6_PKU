import os
import argparse

# augmentation+smoothing+pretrain_cnn+freeze_cnn
lr = 1e-6
training_epochs = 10
name = 'seed615_rl_trainall'
scheduler_decay = 0.98
caption_model_path ='./models/finetune_seed615_trainall/8.pt'
os.system(f'python train_rl.py --lr {lr} --scheduler_decay {scheduler_decay} '
          f'--training_epochs {training_epochs} --name {name} '
          f'--load_pretrain_model --pretrain_model_path {caption_model_path} '
          f'--spec_augmentation  --label_smoothing')