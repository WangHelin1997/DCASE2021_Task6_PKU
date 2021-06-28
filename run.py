import os
import argparse

# augmentation+smoothing+pretrain_cnn+freeze_cnn
lr = 3e-4
training_epochs = 30
name = 'seed1111'
cuda = 5
bs = 32
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--freeze_cnn  --load_pretrain_cnn --batch_size {bs}'
           )

lr = 2e-5
training_epochs = 15
scheduler_decay = 0.98
caption_model_path ='models/seed1111/30.pt'
name = 'finetune_seed1111'
bs = 8
mode = "train"
os.system(f'python train.py --lr {lr} --scheduler_decay {scheduler_decay} '
          f'--training_epochs {training_epochs} --name {name} '
          f'--load_pretrain_model --pretrain_model_path {caption_model_path} '
          f'--spec_augmentation  --label_smoothing '
          f'--batch_size {bs} --mode {mode}'
)

