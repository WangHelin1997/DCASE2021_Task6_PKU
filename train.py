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

hp = hparams()
parser = argparse.ArgumentParser(description='hparams for model')


np.random.seed(hp.seed)
torch.manual_seed(hp.seed)
random.seed(hp.seed)

def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    # return F.binary_cross_entropy(
    #     output_dict, target_dict)
    return F.binary_cross_entropy_with_logits(
        output_dict, target_dict)
def train(epoch, max_epoch, mixup=False, augmentation=None):
    model.train()
    total_loss_text = 0.
    start_time = time.time()
    batch = 0
    return_loss = []
    with torch.autograd.set_detect_anomaly(True):
        for src, tgt, tgt_len, ref, filename in training_data:
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
            output = model(src, tgt_in, epoch, max_epoch)
            # No padding for loss calculation
            if hp.NopadLoss:
                tgt_y_ = torch.cat([tgt_y[j][:tgt_len[j]-1] for j in range(tgt_y.shape[0])], 0)
                output_ = torch.cat([output[j][:tgt_len[j]-1] for j in range(tgt_y.shape[0])], 0)
            else:
                tgt_y_ = tgt_y
                output_ = output

            if mixup:
                loss_text = clip_bce(output_, tgt_y_)
            else:
                loss_text = criterion(output_.contiguous().view(-1, hp.ntoken), tgt_y_.contiguous().view(-1))

            loss = loss_text
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad)
            optimizer.step()
            total_loss_text += loss_text.item()
            return_loss.append(loss_text.item())
            
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
                
                total_loss_text = 0
                start_time = time.time()
    return np.mean(return_loss)

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

def eval_with_beam_csv(evaluation_data, max_len=30, eos_ind=9, word_dict_pickle_path=None, beam_size=3, eval_model='Transformer'):
    model.eval()

    with torch.no_grad():
        with open("test_out.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(['filename','caption_groudtruth','caption_predicted'])
            for src, tgt, _, ref, filename in evaluation_data:
                print(src.shape)
                src = src.to(device)
                tgt = tgt.numpy().tolist()
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
                _ , ref_str = get_eval(tgt, ref, hp.word_dict_pickle_path)
                
                # print(ref_str[0])
                for caption, groundtruth,fname in zip(out_str, ref_str,filename):
                    writer.writerow([fname,groundtruth,caption])
if __name__ == '__main__':
    parser.add_argument('--device', type=str, default=hp.device)
    parser.add_argument('--nlayers', type=int, default=hp.nlayers)
    parser.add_argument('--nhead', type=int, default=hp.nhead)
    parser.add_argument('--nhid', type=int, default=hp.nhid)
    parser.add_argument('--batch_size', type=int, default=hp.batch_size)
    parser.add_argument('--training_epochs', type=int, default=hp.training_epochs)
    parser.add_argument('--lr', type=float, default=hp.lr)
    parser.add_argument('--scheduler_decay', type=float, default=hp.scheduler_decay)
    parser.add_argument('--load_pretrain_cnn', action='store_true')
    parser.add_argument('--freeze_cnn', action='store_true')
    parser.add_argument('--load_pretrain_emb', action='store_true')
    parser.add_argument('--load_pretrain_model', action='store_true')
    parser.add_argument('--spec_augmentation', action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--name', type=str, default=hp.name)
    parser.add_argument('--mode', type=str, default=hp.mode)
    parser.add_argument('--eval_dir', type=str, default=hp.eval_dir)
    parser.add_argument('--pretrain_emb_path', type=str, default=hp.pretrain_emb_path)
    parser.add_argument('--pretrain_cnn_path', type=str, default=hp.pretrain_cnn_path)
    parser.add_argument('--pretrain_model_path', type=str, default=hp.pretrain_model_path)
    parser.add_argument('--Decoder', type=str, default=hp.decoder)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(hp, k, v)
    args = parser.parse_args()

    device = torch.device(hp.device)
    eval_model = hp.decoder
    pretrain_emb = align_word_embedding(hp.word_dict_pickle_path, hp.pretrain_emb_path, hp.ntoken,
                                        hp.emb_size,load_type='bert') if hp.load_pretrain_emb else None
    pretrain_cnn = torch.load(hp.pretrain_cnn_path, map_location="cpu") if hp.load_pretrain_cnn else None
    
    if hp.decoder == 'AttDecoder':
        if hp.load_pretrain_emb:
            print("load pretrain embedding")
        model = AttModel(hp.ninp,hp.nhid,hp.output_dim_encoder,hp.emb_size,hp.dropout_p_encoder,
        hp.output_dim_h_decoder,hp.ntoken,hp.dropout_p_decoder,hp.max_out_t_steps,device,'tag',pretrain_emb,hp.tag_emb,
        hp.multiScale,hp.preword_emb,hp.two_stage_cnn,hp.usingLM).to(device)
        print("The model is", hp.decoder)
        if pretrain_cnn is not None:
            dict_trained = pretrain_cnn
            dict_new = model.encoder.state_dict().copy()
            new_list = list(model.encoder.state_dict().keys())
            trained_list = list(dict_trained.keys())
            for i in range(len(new_list)):
                print(new_list[i])
                dict_new[new_list[i]] = dict_trained[trained_list[i]]
            model.encoder.load_state_dict(dict_new)
            if hp.two_stage_cnn:
                model.encoder_fixed.load_state_dict(dict_new)
    elif hp.decoder == 'Transformer': # no used now
        model = TransformerModel(hp.ntoken, hp.ninp, hp.nhead, hp.nhid, hp.nlayers, hp.batch_size, dropout=0.2,
                             pretrain_cnn=pretrain_cnn, pretrain_emb=pretrain_emb, freeze_cnn=hp.freeze_cnn).to(device)
    else :
        print('exit!!!')
        sys.exit(0)

    if hp.load_pretrain_model:
        model.load_state_dict(torch.load(hp.pretrain_model_path,map_location="cpu"))
    print("freeze_cnn", hp.freeze_cnn)
    if hp.freeze_cnn:
        model.freeze_cnn()
        print("freeze_cnn has finished!")
    if hp.freeze_classifer and hp.two_stage_cnn:
        model.freeze_classifer()
        print("freeze_classifer has finished!")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hp.scheduler_decay)

    if hp.label_smoothing:
        criterion = LabelSmoothingLoss(hp.ntoken, smoothing=0.1, word_freq=hp.word_freq_reciprocal_pickle_path)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=hp.ntoken - 1)
    if hp.multi_gpu:
        device_ids = [4,5]
        # model.to(device)
        model = torch.nn.DataParallel(model,device_ids=device_ids)

    now_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))
    log_dir = 'models/{name}'.format(name=hp.name)

    writer = SummaryWriter(log_dir=log_dir)

    log_path = os.path.join(log_dir, 'train.log')

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
            tr_loss = train(epoch, hp.training_epochs, hp.mixup, mixup_augmentation)
            logging.info('| epoch {:3d} | loss-mean-text {:5.4f}'.format(
                    epoch, tr_loss))
            torch.save(model.state_dict(), '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
            scheduler.step(epoch)
            epoch += 1
        eval_all(validation_beam, word_dict_pickle_path=word_dict_pickle_path, eval_model=eval_model)
        eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                    beam_size=2, eval_model=eval_model)
        eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                    beam_size=3, eval_model=eval_model)
        eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                    beam_size=4, eval_model=eval_model)

        

    if hp.mode == 'eval':
        epoch = 10
        while epoch < hp.training_epochs + 1:
            # Evaluation model score
            logging.info("The epoch is {}".format(epoch))
            model.load_state_dict(torch.load("./models/"+hp.eval_dir+str(epoch)+".pt",map_location="cpu"))
            logging.info(" evaluation ")
            eval_all(validation_beam, word_dict_pickle_path=word_dict_pickle_path, eval_model=eval_model)
            eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                    beam_size=2, eval_model=eval_model)
            eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                        beam_size=3, eval_model=eval_model)
            eval_with_beam(validation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                        beam_size=4, eval_model=eval_model)
            
            epoch += 1

    elif hp.mode == 'test':
        # Generate caption(in test_out.csv)
        model.load_state_dict(torch.load("./models/seed1111_rl_trainall/8.pt",map_location="cpu"))
        test_with_beam(test_data, beam_size=4, eval_model=eval_model,name="seed1111_rl")

