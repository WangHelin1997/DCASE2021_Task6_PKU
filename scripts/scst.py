import numpy as np
import time
import torch
import torch.nn as nn
from .cider import CiderScorer

cider_scorer = CiderScorer(df="scripts/output_pkl_inds.p")
# def init(pickle_dir):
#     cider_scorer = CiderScorer(df="scripts/output_pkl_inds.p")
#     return cider_scorer

def array_to_str(arr):
    out = ''
    
    for i in range(len(arr)):
        if arr[i] == 0:
            continue
        out += str(arr[i]) + ' '
    return out.strip()

def remove_eos(output):
    eos_ind = 9
    output_sentence_ind_batch = []
    for single_sample in output:
        output_sentence_ind = []
        for sym in single_sample:
            if sym == eos_ind: break
            output_sentence_ind.append(sym.item())
        # print(output_sentence_ind)
        output_str = array_to_str(output_sentence_ind)
        # print(output_str)
        output_sentence_ind_batch.append(output_str)
    return output_sentence_ind_batch
def get_self_critical_reward(greedy_res, data_gts, gen_result):
    batch_size = gen_result.shape[0]# batch_size = sample_size * seq_per_img
    gen_result = gen_result.data.cpu().numpy()   # (batch_size, max_len)
    greedy_res = greedy_res.data.cpu().numpy()   # (batch_size, max_len)
    res = []
    
    res.extend(remove_eos(gen_result))
    res.extend(remove_eos(greedy_res))

    gts = []
    # print(data_gts,len(data_gts),len(data_gts[0]),len(data_gts[0][0]))
    
    for i in range(batch_size):
        gts.append([array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))])
    # print(res,type(res[0]))
    for i in range(batch_size*2):
        cider_scorer.cook_append(res[i], gts[i%batch_size])
    
    scores_mean, scores_arr = cider_scorer.compute_score()
    print('Cider scores:', scores_mean)
    
    # print(scores_arr)
    scores = scores_arr[:batch_size] - scores_arr[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    rewards = torch.from_numpy(rewards).float()
    return rewards

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        eos_ind = 9
        pad_ind = 4367
        # print(input,seq.shape)
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = ((seq!=eos_ind) &  (seq!=pad_ind)).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        
        return output

class scst_loss(nn.Module):
    def __init__(self, model):
        super(scst_loss, self).__init__()
        self.model = model
        self.rl_crit = RewardCriterion()

    def forward(self, src, tgt, gts):
        device = src.device
        self.model.eval()
        with torch.no_grad():
            greedy_res, _ = self.model._sample(src, tgt, sample_method='greedy')
        self.model.train()
        gen_result, sample_logprobs = self.model._sample(src, tgt, sample_method='sample')
        # print(greedy_res,gen_result)
        reward = get_self_critical_reward(greedy_res, gts, gen_result)
        reward = torch.from_numpy(reward).float().to(device)
        loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
        return loss