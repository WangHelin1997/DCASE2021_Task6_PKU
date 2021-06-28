import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from data_handling import get_clotho_loader, get_test_data_loader
from model import AttModel # , RNNModel, RNNModelSmall
import itertools
import numpy as np
import os
import sys
import logging
import csv
import random
from util import get_file_list, get_padding, print_hparams, greedy_decode, \
    calculate_bleu, calculate_spider, LabelSmoothingLoss, beam_search, align_word_embedding, gen_str, Mixup, tgt2onehot, do_mixup, get_eval
from hparams import hparams
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.nn import CrossEntropyLoss,BCELoss,BCEWithLogitsLoss
from scripts.scst import scst_loss,RewardCriterion,get_self_critical_reward
hp = hparams()
parser = argparse.ArgumentParser(description='hparams for model')

device = torch.device(hp.device)
np.random.seed(hp.seed)
torch.manual_seed(hp.seed)
def train(epoch, max_epoch, mixup=False, augmentation=None):
    model.train()
    total_loss_text = 0.
    start_time = time.time()
    batch = 0
    with torch.autograd.set_detect_anomaly(True):
        for src, tgt, tgt_len, ref,filename in training_data:
            tgt_y = tgt[:, 1:]
            if mixup:
                mixup_lambda = augmentation.get_lambda(src.shape[0])
                tgt_y = tgt2onehot(tgt_y,hp.ntoken)
                src = do_mixup(src,mixup_lambda=mixup_lambda).float()
                tgt_y = do_mixup(tgt_y,mixup_lambda=mixup_lambda).float()
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_pad_mask = get_padding(tgt, tgt_len)
            tgt_in = tgt[:, :-1]
            tgt_pad_mask = tgt_pad_mask[:, :-1]
            tgt_y = tgt_y.to(device)
            
            optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                greedy_res,_ = model._sample(src, tgt, sample_method='greedy')
                # print(greedy_res)
            model.train()
            gen_result, sample_logprobs = model._sample(src, tgt, sample_method='sample')
            reward = get_self_critical_reward(greedy_res, ref, gen_result)
            loss_text = criterion(sample_logprobs, gen_result, reward.to(device))

            loss = loss_text
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad)
            optimizer.step()
            total_loss_text += loss_text.item()

            writer.add_scalar('Loss/train-text', loss_text.item(), (epoch - 1) * len(training_data) + batch)

            batch += 1

            if batch % hp.log_interval == 0 and batch > 0:
                mean_text_loss = total_loss_text / hp.log_interval
                elapsed = time.time() - start_time
                current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2e} | ms/batch {:5.2f} | '
                            'loss-text {:5.4f}'.format(
                    epoch, batch, len(training_data), current_lr,
                    elapsed * 1000 / hp.log_interval, mean_text_loss))
                # torch.save(model.state_dict(), '{log_dir}/{epoch}_{iterations}.pt'.format(log_dir=log_dir, epoch=epoch,iterations=batch))
                total_loss_text = 0
                start_time = time.time()
def eval_all(evaluation_data, max_len=30, eos_ind=9, word_dict_pickle_path=None, eval_model='Transformer'):
    model.eval()
    
    with torch.no_grad():
        output_sentence_all = []
        ref_all = []
        for src, tgt, _, ref, filename in evaluation_data:
            src = src.to(device)
            if eval_model == 'Transformer':
                output = greedy_decode(model, src, max_len=max_len)
                
            else:
                output = model.greedy_decode(src)

            output_sentence_ind_batch = []
            for i in range(output.size()[0]):
                output_sentence_ind = []
                for j in range(1, output.size(1)):
                    sym = output[i, j]
                    if sym == eos_ind: break
                    output_sentence_ind.append(sym.item())
                output_sentence_ind_batch.append(output_sentence_ind)
            output_sentence_all.extend(output_sentence_ind_batch)
            ref_all.extend(ref)

        score, output_str, ref_str = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)

        loss_mean = score
        writer.add_scalar(f'Loss/eval_greddy', loss_mean, epoch)
        msg = f'eval_greddy SPIDEr: {loss_mean:2.4f}'
        logging.info(msg)
   
def eval_with_beam(evaluation_data, max_len=30, eos_ind=9, word_dict_pickle_path=None, beam_size=3, eval_model='Transformer'):
    model.eval()
    with torch.no_grad():
        output_sentence_all = []
        ref_all = []
        for src, tgt, _, ref,filename in evaluation_data:
            src = src.to(device)

            if eval_model == 'Transformer':
                output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size)
            else:
                output = model.beam_search(src, beam_width=beam_size)
            output_sentence_ind_batch = []
            for single_sample in output:
                output_sentence_ind = []
                for sym in single_sample:
                    if sym == eos_ind: break
                    if eval_model == 'Transformer':
                        output_sentence_ind.append(sym.item())
                    else:
                        output_sentence_ind.append(sym)
                output_sentence_ind_batch.append(output_sentence_ind)
            output_sentence_all.extend(output_sentence_ind_batch)
            ref_all.extend(ref)

        score, output_str, ref_str = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)

        loss_mean = score
        writer.add_scalar(f'Loss/eval_beam', loss_mean, epoch)
        msg = f'eval_beam_{beam_size} SPIDEr: {loss_mean:2.4f}'
        logging.info(msg)


def test_with_beam(test_data, max_len=30, eos_ind=9, beam_size=3,eval_model='Transformer',name="seed1111"):
    model.eval()

    with torch.no_grad():
        save_name  = "test_out_" + name + ".csv"
        with open(save_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'caption_predicted'])
            for src, filename in test_data:
                src = src.to(device)
                if eval_model == 'Transformer':
                    output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size)
                else:
                    output = model.beam_search(src, beam_width=beam_size)
                output_sentence_ind_batch = []
                for single_sample in output:
                    output_sentence_ind = []
                    for sym in single_sample:
                        if sym == eos_ind: break
                        if eval_model == 'Transformer':
                            output_sentence_ind.append(sym.item())
                        else:
                            output_sentence_ind.append(sym)
                    output_sentence_ind_batch.append(output_sentence_ind)
                out_str = gen_str(output_sentence_ind_batch, hp.word_dict_pickle_path)
                for caption, fn in zip(out_str, filename):
                    writer.writerow(['{}.wav'.format(fn), caption])
if __name__ == '__main__':
    parser.add_argument('--device', type=str, default=hp.device)
    parser.add_argument('--nlayers', type=int, default=hp.nlayers)
    parser.add_argument('--nhead', type=int, default=hp.nhead)
    parser.add_argument('--nhid', type=int, default=hp.nhid)
    parser.add_argument('--training_epochs', type=int, default=hp.training_epochs)
    parser.add_argument('--lr', type=float, default=hp.lr)
    parser.add_argument('--scheduler_decay', type=float, default=hp.scheduler_decay)
    parser.add_argument('--load_pretrain_cnn', action='store_true')
    parser.add_argument('--freeze_cnn', action='store_true')
    parser.add_argument('--load_pretrain_emb', action='store_true')
    parser.add_argument('--load_pretrain_model', action='store_true')
    parser.add_argument('--spec_augmentation', action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--name', type=str, default=hp.name)
    parser.add_argument('--pretrain_emb_path', type=str, default=hp.pretrain_emb_path)
    parser.add_argument('--pretrain_cnn_path', type=str, default=hp.pretrain_cnn_path)
    parser.add_argument('--pretrain_model_path', type=str, default=hp.pretrain_model_path)
    parser.add_argument('--Decoder', type=str, default=hp.decoder)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(hp, k, v)
    args = parser.parse_args()
    eval_model = hp.decoder
    pretrain_emb = align_word_embedding(hp.word_dict_pickle_path, hp.pretrain_emb_path, hp.ntoken,
                                        hp.emb_size,load_type='bert') if hp.load_pretrain_emb else None
    pretrain_cnn = torch.load(hp.pretrain_cnn_path, map_location="cpu") if hp.load_pretrain_cnn else None

    model = AttModel(hp.ninp,hp.nhid,hp.output_dim_encoder,hp.emb_size,hp.dropout_p_encoder,
        hp.output_dim_h_decoder,hp.ntoken,hp.dropout_p_decoder,hp.max_out_t_steps,device,'tag',pretrain_emb,hp.tag_emb,
        hp.multiScale,hp.preword_emb,hp.two_stage_cnn,hp.usingLM).to(device)
    print("pretrain model path is",hp.pretrain_model_path)
    model.load_state_dict(torch.load(hp.pretrain_model_path, map_location="cpu"))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hp.scheduler_decay)
    criterion = RewardCriterion()

    now_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))
    log_dir = 'models/{name}'.format(name=hp.name)

    writer = SummaryWriter(log_dir=log_dir)

    log_path = os.path.join(log_dir, 'train.log')

    # device_ids = [4,5,6]
    # # model.to(device)
    # model = torch.nn.DataParallel(model,device_ids=device_ids)

    logging.basicConfig(level=logging.DEBUG,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler(sys.stdout)]
                        )

    data_dir = hp.data_dir
    eval_data_dir = hp.eval_data_dir
    train_data_dir = hp.train_data_dir
    word_dict_pickle_path = hp.word_dict_pickle_path
    word_freq_pickle_path = hp.word_freq_pickle_path
    test_data_dir = hp.test_data_dir
    if hp.train_all:
        training_data = get_clotho_loader(data_dir=data_dir, split='all',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=hp.batch_size,
                                        nb_t_steps_pad='max',
                                        num_workers=4, return_reference=True, augment=hp.spec_augmentation)
    else:
        training_data = get_clotho_loader(data_dir=data_dir, split='development',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=hp.batch_size,
                                        nb_t_steps_pad='max',
                                        num_workers=4, return_reference=True, augment=hp.spec_augmentation)
    validation_beam = get_clotho_loader(data_dir=data_dir, split='validation',
                                    input_field_name='features',
                                    output_field_name='words_ind',
                                    load_into_memory=False,
                                    batch_size=32,
                                    nb_t_steps_pad='max',
                                    shuffle=False,
                                    drop_last=False,
                                    return_reference=True)

    evaluation_beam = get_clotho_loader(data_dir=data_dir, split='evaluation',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=32,
                                        nb_t_steps_pad='max',
                                        shuffle=False,
                                        drop_last=False,
                                        return_reference=True)
    test_data = get_test_data_loader(data_dir=test_data_dir,
                                     batch_size=hp.batch_size * 2,
                                     nb_t_steps_pad='max',
                                     shuffle=False,
                                     drop_last=False,
                                     input_pad_at='start',
                                     num_workers=8)
    logging.info(str(model))

    logging.info(str(print_hparams(hp)))

    logging.info('Data loaded!')
    logging.info('Data size: ' + str(len(training_data)))

    logging.info('Total Model parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    epoch = 1
    print('mixup is ',hp.mixup)
    if hp.mixup:
        mixup_augmentation = Mixup(mixup_alpha=1.0)
    else :
        mixup_augmentation = None
    if hp.mode == 'train':
        while epoch < hp.training_epochs + 1:
            epoch_start_time = time.time()
            train(epoch, hp.training_epochs, hp.mixup, mixup_augmentation)
            torch.save(model.state_dict(), '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
            scheduler.step(epoch)
            eval_all(validation_beam, word_dict_pickle_path=word_dict_pickle_path, eval_model=eval_model)
            eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=2, eval_model=eval_model)
            eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=3, eval_model=eval_model)
            eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=4, eval_model=eval_model)
            epoch += 1
    elif hp.mode == 'eval':
        epoch = 8
        while epoch < hp.training_epochs + 1:
            logging.info("The epoch is {}".format(epoch))
            model.load_state_dict(torch.load("./models/"+hp.eval_dir+str(epoch)+".pt",map_location="cpu"))
            eval_all(evaluation_beam, word_dict_pickle_path=word_dict_pickle_path, eval_model=eval_model)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=2, eval_model=eval_model)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=3, eval_model=eval_model)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=4, eval_model=eval_model)
            epoch += 1
    

        


