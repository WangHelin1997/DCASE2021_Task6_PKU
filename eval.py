import os
import argparse

# evaluation
# training_epochs = 30
# name = 'eval_fine_tune_aoa_gru'
# # name = "eval_aoa_normal"
# os.system(f'python train.py '
#           f' --training_epochs {training_epochs} '
#           f'--name {name} '
#            )

# training_epochs = 30
# name = 'eval_baseline'
# # name = "eval_aoa_normal"
# bs = 32
# mode = "eval"
# eval_dir = "baseline/"
# os.system(f'python train.py '
#           f' --training_epochs {training_epochs} '
#           f'--name {name} '
#           f'--batch_size {bs} --mode {mode} --eval_dir {eval_dir}'
#            )

training_epochs = 10
name = 'eval_test_1'
# name = "eval_aoa_normal"
bs = 32
mode = "eval"
eval_dir = "seed1111_rl/"
os.system(f'python train.py '
          f' --training_epochs {training_epochs} '
          f'--name {name} '
          f'--batch_size {bs} --mode {mode} --eval_dir {eval_dir}'
           )