import os
import argparse

#fine-tune
lr = 2e-5
training_epochs = 15
scheduler_decay = 0.98
caption_model_path ='models/seed615_trainall/30.pt'
name = 'finetune_seed615_trainall'
bs = 8
mode = "train"
os.system(f'python train.py --lr {lr} --scheduler_decay {scheduler_decay} '
          f'--training_epochs {training_epochs} --name {name} '
          f'--load_pretrain_model --pretrain_model_path {caption_model_path} '
          f'--spec_augmentation  --label_smoothing '
          f'--batch_size {bs} --mode {mode}'
          )

# training_epochs = 15
# name = 'eval_finetune_baseline_29'
# # name = "eval_aoa_normal"
# bs = 32
# mode = "eval"
# eval_dir = "finetune_baseline_29/"
# os.system(f'python train.py '
#           f' --training_epochs {training_epochs} '
#           f'--name {name} '
#           f'--batch_size {bs} --mode {mode} --eval_dir {eval_dir}'
#            )