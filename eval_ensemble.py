import os
import argparse

# evaluation
training_epochs = 11
name = 'ensemble_baseline'

os.system(f'python ensemble.py '
          f' --training_epochs {training_epochs} '
          f'--name {name} '
           )
