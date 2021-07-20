# PKU Team Code for DCASE 2021 Task6

This is the code of PKU team for DCASE 2021 Task 6. 
### Setting up the Code and Environment
1. Clone this repository: `https://github.com/WangHelin1997/DCASE2021_Task6_PKU.git`
2. Install pytorch >=1.4.0
3. Use pip to install dependencies: `pip install -r requirements.txt`
### Preparing the data
+ [Download](http://dcase.community/challenge2020/task-automatic-audio-captioning#download) the Clotho dataset for DCASE2021 Automated Audio Captioning challenge. And how to prepare training data and setup coco caption, please refer to [Dcase2020 BUPT team's](https://github.com/lukewys/dcase_2020_T6)
+ Enter the **audio_tag** directory. 
+ Firstly, run `python generate_word_list.py` to create words list `word_list_pretrain_rules.p` and tagging words to indexes of embedding layer `TaggingtoEmbs`. 
+ Then run `python generate_tag.py` to generate `audioTagName_{development/validation/evaluation}_fin_nv.pickle` and `audioTagNum_{development/validation/evaluation}_fin_nv.pickle` 

### Configuration
The training configuration is saved in the `hparams.py` and you can reset it to your own parameters.    
### Training tagging model
Run the `Tag_train.py`. Firstly, train the tagging model by freezing up the CNN for 80 epochs, then fintune it for 25 epochs. Finally, the mAP of tagging could reach 0.287 in the evaluation splits.

### Training the captioning model 
+ We choose the 25th epoch keyword pre-trained model for our final encoder.
#### Train baseline model
+ Run `python run.py`, it will freeze up the encoder and just train the part of decoder for 30 epochs. We choose the best model in validation splits for the next step training.
+ The scores of validation splits will be shown after every epoch.

#### Train baseline model by optimizing CIDEr
+ Run `python run_rl.py` to train the model by opytimizing CIDEr.
+ The scores of validation splits will be shown after every epoch.

### Eval 
+ Run `python eval.py` to get the score of a single model.
+ Run `python eval_ensemble.py` to get the score of an ensemble model.
    + Modify `eval_ensemble.py` to ensemble models by epochs.
    + Or Modify `ensemble.py` to select models of different seeds.
### Test
Set `mode=test` in `hparams.py`. Then run `python train.py`, `python train_rl.py` or `python ensemble.py` to get the final results in test splits.





