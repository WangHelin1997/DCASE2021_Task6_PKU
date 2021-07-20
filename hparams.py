from pathlib import Path


class hparams:
    batch_size = 32
    nhid = 256
    output_dim_encoder = 512
    output_dim_h_decoder = 512
    att_size = 512
    emb_size = 512
    dropout_p_encoder = 0.25
    dropout_p_decoder = 0.25
    max_out_t_steps = 30

    nhead = 4
    natt = 512
    nlayers = 2
    ninp = 64
    ntoken = 4367 + 1
    clip_grad = 2.5
    lr = 3e-4  # learning rate
    beam_width = 3
    training_epochs = 50
    log_interval = 100
    checkpoint_save_interval = 5
    decoder = 'AttDecoder'

    seed = 1111
    device = 'cuda:1'   #'cuda:0' 'cuda:1' 'cpu'
    mode = 'train'
    name = 'base'
    nkeyword = 4979
    # augmentation mixup
    mixup = False
    label_smoothing = True
    load_pretrain_cnn = True
    # freeze_cnn = True
    freeze_classifer = True
    load_pretrain_emb = False
    load_pretrain_model = False
    spec_augmentation = True
    scheduler_decay = 0.98
    tmp_name = 'one_stage'
    NopadLoss = True
    tag_emb = True
    multiScale = False
    preword_emb = True
    two_stage_cnn = False
    usingLM = False
    multi_gpu = False
    train_all = False
# Train tagging model
    train_tag = False
    save_name = 'tag_models_baseline'
    class_num = 300
    tag_mixup  = True
    tag_focalLoss = False
    tag_GMAP = False
    tag_specMix = True
# data(default)
    data_dir = Path(r'../DCASE2021_Task6/create_dataset/data/data_splits')
    eval_data_dir = r'../DCASE2021_Task6/create_dataset/data/data_splits/evaluation'
    train_data_dir = r'./DCASE2021_Task6/create_dataset/data/data_splits/development'
    test_data_dir = r'./DCASE2021_Task6/create_dataset/data/test_data'
    word_dict_pickle_path = r'../DCASE2021_Task6/create_dataset/data/pickles/words_list.p'
    word_freq_pickle_path = r'../DCASE2021_Task6/create_dataset/data/pickles/words_frequencies.p'
    word_freq_reciprocal_pickle_path = r'../DCASE2021_Task6/create_dataset/data/pickles/words_weight.pickle'
    # pretrain_model
    tag_keyword_pickle_path = r'./audio_tag/word_list_pretrain_rules.p'
    tagging_to_embs = r'./audio_tag/TaggingToEmbs.p'
    pretrain_emb_path = './create_dataset/data/pickles/words_list_glove.p'
    #pretrain_emb_path = r'./bert_last_hidden.pickle'
    # pretrain_cnn_path = r'./models/tag_models_baseline_finetune/TagModel_25.pt'
    pretrain_cnn_path = r'../DCASE2021_Task6/models/tag_models_baseline_finetune/TagModel_25.pt'
    pretrain_model_path = r'models/baseline/30.pt'
    #eval dir
    eval_dir = "seed1111/"


