import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from data_handling import get_clotho_loader, get_test_data_loader
from model import *  # , RNNModel, RNNModelSmall
import itertools
import numpy as np
import os
import sys
import logging
import csv

from util import get_file_list, get_padding, print_hparams, greedy_decode, \
    calculate_bleu, calculate_spider, LabelSmoothingLoss, beam_search, align_word_embedding, gen_str, Mixup, tgt2onehot, \
    do_mixup, get_eval
from hparams import hparams
from torch.utils.tensorboard import SummaryWriter
import argparse

hp = hparams()
parser = argparse.ArgumentParser(description='hparams for model')

device = torch.device(hp.device)
np.random.seed(hp.seed)
torch.manual_seed(hp.seed)


def eval_all(evaluation_data, ensemble_model, max_len=30, eos_ind=9, word_dict_pickle_path=None):

    with torch.no_grad():
        output_sentence_all = []
        ref_all = []
        for src, tgt, _, ref, filename in evaluation_data:
            src = src.to(device)
            output = ensemble_model.greedy_decode(src)

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


def eval_with_beam(evaluation_data, ensemble_model, max_len=30, eos_ind=9, word_dict_pickle_path=None, beam_size=3):
    with torch.no_grad():
        output_sentence_all = []
        ref_all = []
        for src, tgt, _, ref, filename in evaluation_data:
            src = src.to(device)

            output = ensemble_model.beam_search(src, beam_width=beam_size)
            output_sentence_ind_batch = []
            for single_sample in output:
                output_sentence_ind = []
                for sym in single_sample:
                    if sym == eos_ind: break
                    output_sentence_ind.append(sym)
                output_sentence_ind_batch.append(output_sentence_ind)
            output_sentence_all.extend(output_sentence_ind_batch)
            ref_all.extend(ref)

        score, output_str, ref_str = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)

        loss_mean = score
        writer.add_scalar(f'Loss/eval_beam', loss_mean, epoch)
        msg = f'eval_beam_{beam_size} SPIDEr: {loss_mean:2.4f}'
        logging.info(msg)


def test_with_beam(test_data, ensemble_model,max_len=30, eos_ind=9, beam_size=3,eval_model='Transformer',name="seed1111"):
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
                    output = ensemble_model.beam_search(src, beam_width=beam_size)
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


def eval_with_beam_csv(evaluation_data, max_len=30, eos_ind=9, word_dict_pickle_path=None, beam_size=3,
                       eval_model='Transformer'):
    model.eval()

    with torch.no_grad():
        with open("test_out.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'caption_groudtruth', 'caption_predicted'])
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
                _, ref_str = get_eval(tgt, ref, hp.word_dict_pickle_path)

                # print(ref_str[0])
                for caption, groundtruth, fname in zip(out_str, ref_str, filename):
                    writer.writerow([fname, groundtruth, caption])


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

    evaluation_beam = get_clotho_loader(data_dir=data_dir, split='evaluation',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=16,
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

    logging.info(str(print_hparams(hp)))

    logging.info('Data loaded!')
    logging.info('Data size: ' + str(len(training_data)))
    epoch = 10

    if hp.mode == 'eval':
        model_list=[]    
        # ensemble models by epochs
        while epoch < hp.training_epochs + 1:
            model = AttModel(hp.ninp, hp.nhid, hp.output_dim_encoder, hp.emb_size, hp.dropout_p_encoder,
                             hp.output_dim_h_decoder, hp.ntoken, hp.dropout_p_decoder, hp.max_out_t_steps, device,
                             'tag', None, hp.tag_emb, hp.multiScale, hp.preword_emb, hp.two_stage_cnn, hp.usingLM).to(device)
            model.load_state_dict(torch.load("./models/finetune_seed1111/" + str(epoch) + ".pt",map_location='cpu'))
            model.eval()
            model_list.append(model)
            epoch += 1
       
        ensemble_model = EnsembleModel(model_list)
        eval_with_beam(evaluation_beam, ensemble_model, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                       beam_size=4)
    elif hp.mode == 'test':
        # Generate caption(in test_out.csv)
        model_list=[]
        model_name = ["./models/finetune_seed1111_trainall/8.pt","./models/finetune_seed6666_trainall/8.pt","./models/finetune_seed615_trainall/8.pt",
        "./models/finetune_seed1111_trainall/9.pt","./models/finetune_seed6666_trainall/9.pt","./models/finetune_seed615_trainall/9.pt",
        "./models/finetune_seed1111_trainall/10.pt","./models/finetune_seed6666_trainall/10.pt","./models/finetune_seed615_trainall/10.pt"]
        for name in model_name:
            model = AttModel(hp.ninp, hp.nhid, hp.output_dim_encoder, hp.emb_size, hp.dropout_p_encoder,
                            hp.output_dim_h_decoder, hp.ntoken, hp.dropout_p_decoder, hp.max_out_t_steps, device,
                            'tag', None, hp.tag_emb, hp.multiScale, hp.preword_emb, hp.two_stage_cnn, hp.usingLM).to(device)
            model.load_state_dict(torch.load(name,map_location='cpu'))
            model.eval()
            model_list.append(model)
        ensemble_model = EnsembleModel(model_list)
        test_with_beam(test_data, ensemble_model, beam_size=4, eval_model=eval_model,name="seed11116666615_ensemble_epoch8_9_10")
    else:
        raise ValueError("Mode must be eval or test!")

