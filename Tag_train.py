import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn import metrics
from audio_tag.data_loader import tag_loader
import numpy as np
from hparams import hparams as hp
from encoder import Tag
from tqdm import tqdm
from util import Mixup
import sys
import os
import logging
from loss import FocalLoss
import random
from hparams import hparams

# initial setting
hp  = hparams()
random.seed(hp.seed)
np.random.seed(hp.seed)
torch.manual_seed(hp.seed)

mixup  = hp.tag_mixup
focalLoss = hp.tag_focalLoss
GMAP = hp.tag_GMAP
specMix = hp.tag_specMix
Train_ALL = hp.train_all
train_b = hp.train_tag
filename = hp.save_name
class_num = hp.class_num
device = torch.device(hp.device)
data_dir = hp.data_dir
finetune = not train_b
freeze_cnn= train_b

if freeze_cnn:
    learning_rate = 1e-3
    epoch_num = 100
    batch_size_train = 64
    batch_size_val = 32
else:
    learning_rate = 5e-4
    epoch_num = 25
    batch_size_train = 16
    batch_size_val = 32
model = Tag(class_num,model_type='resnet38',pretrain_model_path="./models/ResNet38_mAP=0.434.pth",freeze_cnn=freeze_cnn,GMAP=GMAP,specMix=specMix).to(device)
if Train_ALL == True:
    training_data = tag_loader(data_dir=data_dir,split='all',
                                        batch_size=batch_size_train,class_num=class_num)
else:
    training_data = tag_loader(data_dir=data_dir,split='development',
                                        batch_size=batch_size_train,class_num=class_num)

validation_data = tag_loader(data_dir=data_dir,split='validation',
                               batch_size=batch_size_val,class_num=class_num)                               
test_data = tag_loader(data_dir=data_dir,split='evaluation',
                               batch_size=16,class_num=class_num)
optimizer =torch.optim.Adam(model.parameters(), lr=learning_rate,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

def do_mixup(x, mixup_lambda):
    mixup_lambda = torch.tensor(mixup_lambda)
    out = (x[0 :: 2].transpose(0,-1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0,-1) * mixup_lambda[1 :: 2]).transpose(0,-1)
    return out

def clip_bce(output_dict, target_dict):
    return F.binary_cross_entropy(
        output_dict, target_dict)

if mixup:
    mixup_augmentation = Mixup(mixup_alpha=1.0)
    if focalLoss:
        tag_loss = FocalLoss()
    else:
        tag_loss = clip_bce
    print("try mixup augmentation")
else:
    tag_loss = nn.BCELoss()

def train(epoch):
    loss_list = []
    model.train()
    with tqdm(training_data,total=len(training_data)) as bar:
        for i, (feature, tag) in enumerate(bar):
            if mixup:
                feature = do_mixup(feature,mixup_lambda=mixup_augmentation.get_lambda(feature.shape[0])).float()
                tag = do_mixup(tag,mixup_lambda=mixup_augmentation.get_lambda(tag.shape[0])).float()

            feature = feature.to(device)
            tag = tag.to(device)

            optimizer.zero_grad()
            out_tag,o,_,_ = model(feature)
            
            loss = tag_loss(out_tag,tag)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch, i, np.mean(loss_list)))
    return np.mean(loss_list)

def test(epoch,dataset):
    eva_loss = []
    model.eval()
    pred_output = []
    all_tag = []
    with torch.no_grad():
        for i, (feature, tag) in enumerate(dataset):
            feature = feature.to(device)
            tag = tag.to(device)
            out_tag,_,_,_ = model(feature)
            out = out_tag.cpu().numpy()
            tag_ = tag.cpu().numpy()

            pred_output.extend(out)
            all_tag.extend(tag_)
            loss = tag_loss(out_tag, tag)
            eva_loss.append(loss.item())
    
    mean_loss = np.mean(eva_loss)
    pred_output = np.array(pred_output)
    all_tag = np.array(all_tag)

    average_precision = metrics.average_precision_score(
            all_tag, pred_output, average=None)
    msg = "Epoch: {0}  the mAP is {1}".format(epoch, np.mean(average_precision))
    logging.info(msg)
    print("epoch:{:d}--testloss:{:.6f}".format(epoch, mean_loss.item()))
    print('Validate bal mAP: {:.3f}'.format(np.mean(average_precision)))

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob,pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point


if __name__ == '__main__':
    epoch_last = 0
    if Train_ALL == True:
        filename_train = './models/' + filename + "_all"
        filename_finetune = './models/' + filename + '_finetune_all'
    else:
        filename_train = './models/' + filename
        filename_finetune = './models/' + filename + '_finetune'
    if finetune:
        log_dir = filename_finetune
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        log_dir = filename_train
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'train.log')
    logging.basicConfig(level=logging.DEBUG,
                            format=
                            '%(asctime)s - %(levelname)s: %(message)s',
                            handlers=[
                                logging.FileHandler(log_path),
                                logging.StreamHandler(sys.stdout)]
                            )
    if train_b:
        for epoch in range(epoch_last, epoch_last + epoch_num + 1):
            print(optimizer.param_groups[0]['lr'])
            tr_loss = train(epoch)
            logging.info("train loss is {}".format(tr_loss))
            scheduler.step(epoch)
            if not Train_ALL:
                test(epoch,dataset=validation_data)
            if epoch % 5 == 0:
                torch.save(model.state_dict(), filename_train + '/TagModel_{}.pt'.format(epoch))
    if finetune:
        model.load_state_dict(torch.load(filename_train + "/TagModel_{}.pt".format(str(80)),map_location="cpu"))
        print("load model is ",filename_train + "/TagModel_{}.pt".format(str(80)))
        for epoch in range(epoch_last, epoch_last + epoch_num + 1):
            print(optimizer.param_groups[0]['lr'])
            tr_loss = train(epoch)
            logging.info("train loss is {}".format(tr_loss))
            scheduler.step(epoch)
            if not Train_ALL:
                test(epoch,dataset=validation_data)
            if epoch % 5 == 0:
                torch.save(model.state_dict(), filename_finetune + '/TagModel_{}.pt'.format(epoch))
    # Test model for every 5 epochs
    # for epoch in range(0,26,5):
    #     model.load_state_dict(torch.load(filename_finetune + "/TagModel_{}.pt".format(epoch),map_location="cpu"))
    #     test(epoch,dataset=test_data)




