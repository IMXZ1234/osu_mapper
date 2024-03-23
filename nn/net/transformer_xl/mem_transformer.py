import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .log_uniform_sampler import LogUniformSampler, sample_logits


def sample_step(logits, former_tokens, rand_inst: random.Random, p_thresh=0.8):
    """
    nucleus sampling by default
    logits: [L, N, C]
    former_tokens: long tensor [L', N], or None

    return: [L, N]
    """
    # for every autoregressive step within the same segment,
    # yield discrete token from output logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    # print(sorted_probs, indices)

    all_step_sampled_tokens = []
    for step_idx in range(logits.shape[0]):
        sampled_tokens = []
        step_sorted_probs = sorted_probs[step_idx]
        step_indices = indices[step_idx]
        for sample_idx in range(logits.shape[1]):
            sample_sorted_probs = step_sorted_probs[sample_idx]
            summed_p = 0
            for i, p in enumerate(sample_sorted_probs):
                summed_p += p.item()
                if summed_p > p_thresh:
                    break
            reweighted_probs = sample_sorted_probs[:i+1] / summed_p
            # print(summed_p, reweighted_probs)
            sampled_tokens.append(
                # returns a list
                rand_inst.choices(step_indices[sample_idx, :i+1], weights=reweighted_probs, k=1)[0]
            )
        all_step_sampled_tokens.append(sampled_tokens)
    return torch.tensor(all_step_sampled_tokens, dtype=torch.long, device=logits.device)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        # demb = d_model
        self.demb = demb

        # vec [d_emb // 2]
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        # outer product (vec x vec -> mat)
        # [klen, d_model // 2]
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        # [klen, d_model]
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            # [klen, bsz, d_model]
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            # [klen, 1, d_model]
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False, **kwargs):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        # x: [qlen, klen, bsz, n_head]
        # zero_pad: [qlen, 1, bsz, n_head]
        # if we view the matrix x in its underlying linear format,
        # during following process, first qlen items are thrown away,
        # while each line in the 2d format is shifted left by 1 due to head-padded zeros (1 zero each line)
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        # -> [klen+1, qlen, ...]
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    """
    Now can deal with cross-attention by providing 'context'
    """
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, context=None,
                precomputed_k=None, precomputed_v=None):
        """
        core_out        [qlen, bsz, d_model]            w
        pos_emb         [rlen, 1, d_model]              r               rlen = qlen + mlen
        self.r_w_bias   [n_head, d_head]                r_w_bias        learnable parameter, same for all pos & all layer
        self.r_r_bias   [n_head, d_head]                r_r_bias        learnable parameter, same for all pos & all layer
        dec_attn_mask   [qlen, rlen, 1]                 attn_mask
        mems            [mlen, bsz, d_model]                            for a single layer
        context         [qlen, bsz, d_model]                            for a single layer
        precomputed_k   [klen, bsz, n_head, d_head]                     for a single layer, klen = qlen (cross-att) or (qlen + mlen) (self-att)
        precomputed_v   [klen, bsz, n_head, d_head]                     for a single layer

        for self-att:
        w: qlen
        r: klen = mlen + qlen
        dec_attn_mask: klen
        mems: mlen

        for cross-att:
        w: qlen
        r: qlen + qlen
        dec_attn_mask: None
        mems: None
        """
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        # print('w', w.shape, 'r', r.shape, 'mem', mems.shape if mems is not None else None)

        # relative encoding, length of klen
        r_head_k = self.r_net(r)

        use_precomputed_kv = True
        const_precomputed_kv = True
        # if context exists, calculate k, v with context (cross-att),
        # also, for cross-att, k,v can always be computed once and then cached
        if context is None:
            context = w
            const_precomputed_kv = False
            cross_att = False
        else:
            cross_att = True

        if mems is not None:
            # mlen + qlen
            cat = torch.cat([mems, context], 0)
        else:
            # qlen
            cat = context

        if self.pre_lnorm:
            cat = self.layer_norm(cat)
            # if using pre_lnorm in self-att, k, v from former autoregressive steps may be changed
            # by later tokens due to lnorm
            if not const_precomputed_kv:
                use_precomputed_kv = False

        # note k, v (before current step, including those calculated from mem)
        # may be shared across different autoregressive steps in a single segment,
        # and therefore may be reused
        if use_precomputed_kv and (precomputed_k is not None) and (precomputed_v is not None):
            w_head_k, w_head_v = precomputed_k, precomputed_v
        else:
            # klen
            w_head_k, w_head_v = torch.chunk(self.kv_net(cat), 2, dim=-1)
        w_head_q = self.q_net(cat[-qlen:])

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        # print('w_head_q', w_head_q.shape, 'w_head_k', w_head_k.shape, 'w_head_v', w_head_v.shape)

        r_head_k = r_head_k.view(-1, self.n_head, self.d_head)                # klen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        # print('rr_head_q', rr_head_q.shape, 'r_head_k', r_head_k.shape, cross_att)
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        # print('BD.shape', BD.shape)
        # during step-by-step inference, BD is the same as before _rel_shift since qlen=1
        # if not cross_att:
        #     # self-att
        #     # lower half triangle of BD[klen:] contains the desired value,
        #     # while the upper triangle contains values going to be masked out
        #     BD = self._rel_shift(BD)
        # else:
        #     # cross-att
        #     # now we not only have i-j > 0 but also i-j < 0 (attending to future tokens) in relative pose_emb,
        #     # we are reusing original pose_emb where i-j > qlen for positions where i-j < 0
        #     # as long as mlen >= qlen, this strategy should be all right
        #     BD = self._rel_shift(torch.cat([BD, BD], dim=0))[:qlen]
        BD = self._rel_shift(BD).contiguous()[:, :klen]
        # print('after shift BD.shape', BD.shape)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        # print('attn_score.shape', attn_score.shape)
        # print('attn_mask.shape', attn_mask.shape if attn_mask is not None else None)

        #### compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score = torch.masked_fill(attn_score,
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                # by default
                # copy to each attention head
                # where attn_mask is 1, fill with -inf
                attn_score = torch.masked_fill(attn_score,
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)
        # print('after masking', attn_score.shape, attn_score.dtype, attn_score.device)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        # print('after softmax', attn_prob.shape)
        attn_prob = self.dropatt(attn_prob)
        # print('after dropatt', attn_prob.shape)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        # print('after att cal', attn_vec.shape)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection
        output = w + attn_out
        if not self.pre_lnorm:
            # layer normalization
            output = self.layer_norm(output)

        return output


class RelPartialLearnableDecoderLayerWithCrossAtt(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super().__init__()
        self.self_att = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.context_cross_att = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, context=None):
        # r: mlen + qlen + qlen
        # note residual is within each attention block
        # for cross-att, full context is visible to token at every autoregressive step
        qlen, mlen = w.shape[0], mems.shape[0] if mems is not None else 0
        w = self.self_att(w, r[:qlen + mlen], r_w_bias, r_r_bias, attn_mask, mems, None)
        # context can be of length qlen(!=rlen) if context outside current segment is not used
        # r = r[-context.shape[0]:]
        w = self.context_cross_att(w, r[-(qlen+qlen):], r_w_bias, r_r_bias, None, None, context)
        w = self.pos_ff(w)
        return w


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    """
    Used by default
    """
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class ContextEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, length_compress=16, block_compress=4):
        # downsample length_compress
        super(ContextEncoder, self).__init__()
        self.ln = nn.LayerNorm(in_channels)
        self.net = []
        current_compress = length_compress
        current_in_channels = in_channels
        while current_compress > block_compress:
            self.net.extend([
                nn.Conv1d(current_in_channels, out_channels, kernel_size=block_compress * 2 - 1, stride=block_compress, padding=block_compress-1),
                nn.BatchNorm1d(out_channels),
                nn.SiLU(),
            ])
            current_compress = current_compress // block_compress
            current_in_channels = out_channels
        self.net.extend([
            nn.Conv1d(current_in_channels, out_channels, kernel_size=block_compress * 2 - 1, stride=current_compress, padding=block_compress - 1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        ])
        self.net = nn.Sequential(*self.net)

    def forward(self, context):
        """
        N, C, L
        """
        context = self.ln(context.permute(0, 2, 1)).permute(0, 2, 1)
        return self.net(context)


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1, context_compress=16, d_context=40,
                 d_continuous=2, ):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.d_continuous = d_continuous

        self.word_emb = AdaptiveEmbedding(n_token, d_embed - d_continuous, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayerWithCrossAtt(
                        n_head, d_model, d_head, d_inner, dropout,
                        # tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.out_layer = nn.Linear(d_model, n_token, bias=False)
            if tie_weight:
                self.out_layer.weight = self.word_emb.emb_layers[0].weight
        if d_continuous > 0:
            self.out_layer_continuous = nn.Linear(d_model, d_continuous)
            # self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
            #                                         cutoffs, div_val=div_val)
            #
            # if tie_weight:
            #     for i in range(len(self.crit.out_layers)):
            #         self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight
            #
            # if tie_projs:
            #     for i, tie_proj in enumerate(tie_projs):
            #         if tie_proj and div_val == 1 and d_model != d_embed:
            #             self.crit.out_projs[i] = self.word_emb.emb_projs[0]
            #         elif tie_proj and div_val != 1:
            #             self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

        self.context_encoder = ContextEncoder(d_context, d_model, context_compress)

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def init_mem_context(self):
        if self.mem_len > 0:
            param = next(self.parameters())
            return torch.empty(0, dtype=param.dtype, device=param.device)
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _update_mem_context(self, context, mem_context, qlen, mlen):
        # does not deal with None
        if mem_context is None:
            return None

        with torch.no_grad():
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            mem_context = torch.cat([mem_context, context], dim=0)[beg_idx:end_idx].detach()

        return mem_context

    def forward(self, inp_discrete, inp_continuous, context, mems=None):
        """
        training forward

        [N, L], [N, C, L], [N, C, L]
        """
        if not mems:
            mems = self.init_mems()
            # mem_context = self.init_mem_context()

        context = self.context_encoder(context)
        # N, C, L -> L, N, C
        context = context.permute(2, 0, 1)
        # context.shape: [qlen, bsz, d_context]
        # new_mem_context = self._update_mem_context(context, mem_context, qlen, mlen)

        inp_discrete = inp_discrete.permute(1, 0)
        word_emb = self.word_emb(inp_discrete)

        if inp_continuous is not None:
            inp_continuous = inp_continuous.permute(2, 0, 1)
            word_emb = torch.cat([word_emb, inp_continuous], dim=2)

        qlen, bsz = inp_discrete.size()
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        all_ones = torch.ones([qlen, klen], dtype=torch.long, device=word_emb.device)
        if self.same_length:
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen) + torch.tril(all_ones, -mask_shift_len))
        else:
            dec_attn_mask = torch.triu(all_ones, diagonal=1+mlen)
        dec_attn_mask = dec_attn_mask[:, :, None]
        dec_attn_mask = (dec_attn_mask == 1)

        hids = []
        if self.attn_type == 0: # default
            # of length qlen + mlen
            # pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
            #                        dtype=word_emb.dtype)
            pos_seq = torch.arange(klen-1, -qlen-1, -1, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                # shared r_w_bias r_r_bias
                # currently we are not going to use context from former segments
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, attn_mask=dec_attn_mask, mems=mems_i, context=context)
                hids.append(core_out)

        core_out = self.drop(core_out)
        # [L, N, C] -> [N, C, L]
        logits = self.out_layer(core_out).permute(1, 2, 0)
        if inp_continuous is not None:
            out_continuous = self.out_layer_continuous(core_out).permute(1, 2, 0)
        else:
            out_continuous = None

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return logits, out_continuous, new_mems

    def forward_segment(self, inp_discrete, inp_continuous, context, mems=None, sample_func=sample_step, rand_inst=None):
        """
        inference, hidden is reused for every autoregressive step within the same segment
        inp_discrete: [N, 1]

        note returns
        """
        # print('inp_discrete', inp_discrete.shape, 'inp_continuous', inp_continuous.shape if inp_continuous is not None else None, 'context', context.shape)
        if rand_inst is None:
            rand_inst = random.Random()

        if not mems:
            mems = self.init_mems()

        context = self.context_encoder(context)
        # N, C, Lc -> Lc, N, C
        context = context.permute(2, 0, 1)

        # first token should be given
        # N, L -> L, N
        inp_discrete = inp_discrete.permute(1, 0)
        # context.shape: [tgt_len, bsz, d_context]
        # assert context.shape[0] == self.tgt_len
        word_emb = self.word_emb(inp_discrete)

        if inp_continuous is not None:
            inp_continuous = inp_continuous.permute(2, 0, 1)
            word_emb = torch.cat([word_emb, inp_continuous], dim=2)

        qlen, bsz = inp_discrete.size()
        mlen = mems[0].size(0) if mems is not None else 0
        tgt_len = context.shape[0]
        klen = mlen + tgt_len
        all_ones = torch.ones([tgt_len, klen], dtype=torch.long, device=word_emb.device)
        if self.same_length:
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = tgt_len - mask_len
            else:
                mask_shift_len = tgt_len
            dec_attn_mask = (torch.triu(all_ones, 1+mlen) + torch.tril(all_ones, -mask_shift_len))
        else:
            dec_attn_mask = torch.triu(all_ones, diagonal=1+mlen)
        dec_attn_mask = dec_attn_mask[:, :, None]
        dec_attn_mask = (dec_attn_mask == 1)

        pos_seq = torch.arange(klen - 1, -tgt_len-1, -1, device=word_emb.device,
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        hids = []
        # list of [L, N], [L, N, C]
        out_discrete, out_continuous = [inp_discrete], [inp_continuous]
        out_logits = []
        core_out = word_emb.new_zeros(tgt_len, bsz, self.d_embed)
        core_out[:qlen] = word_emb
        # inefficient implementation
        for step_idx in range(qlen, tgt_len):
            # autoregressive step prediction
            # len(step_word_emb) always = 1
            # assume all samples within batch is at the same autoregressive step
            if self.attn_type == 0: # default

                core_out = self.drop(core_out)
                step_pos_emb = self.drop(pos_emb)

                if step_idx == tgt_len - 1:
                    # only record hidden (for mem update) at the last autoregressive step within segment
                    hids.append(core_out)
                for i, layer in enumerate(self.layers):
                    mems_i = None if mems is None else mems[i]
                    # shared r_w_bias r_r_bias
                    core_out = layer(core_out, step_pos_emb, self.r_w_bias,
                            self.r_r_bias, attn_mask=dec_attn_mask, mems=mems_i, context=context)
                    if step_idx == tgt_len - 1:
                        hids.append(core_out)

            # extract step output
            step_core_out = core_out[step_idx:step_idx+1, ...]
            step_core_out = self.drop(step_core_out)

            # [L=1, N, C]
            logits = self.out_layer(step_core_out)
            out_logits.append(logits)
            # -> [L=1, N]
            step_out_discrete = sample_func(
                logits,
                torch.cat(out_discrete, dim=0) if len(out_discrete) < 0 else None,
                rand_inst)
            # print('step_out_discrete', step_out_discrete.shape)
            out_discrete.append(step_out_discrete)

            # -> [L=1, N, d_embed-d_continuous]
            step_word_emb = self.word_emb(step_out_discrete)

            if inp_continuous is not None:
                step_out_continuous = self.out_layer_continuous(step_core_out)
                step_word_emb = torch.cat([step_word_emb, step_out_continuous], dim=2)
                out_continuous.append(step_out_continuous)

            # fill in current step output (after sampling and re-embedding)
            if step_idx < tgt_len-1:
                core_out[step_idx+1] = step_word_emb[0]

        new_mems = self._update_mems(hids, mems, tgt_len, mlen)
        # out_logits = torch.cat(out_logits, dim=0).reshape([-1, self.n_token])
        # out_soft = torch.softmax(out_logits, dim=-1).detach().cpu().numpy()
        # print('logits',
        #       np.min(out_soft, axis=0),
        #       np.mean(out_soft, axis=0),
        #       np.max(out_soft, axis=0),
        #       np.std(out_soft, axis=0),
        #       )

        # list of [1, N] -> [N, L+1]
        out_discrete = torch.cat(out_discrete, dim=0).permute(1, 0)
        out_discrete, out_last_discrete = out_discrete[..., :-1], out_discrete[..., -1:]
        if inp_continuous is not None:
            # L+1, N, C -> N, C, L+1
            out_continuous = torch.cat(out_continuous, dim=0).permute(1, 2, 0)
            out_continuous, out_last_continuous = out_continuous[..., :-1], out_continuous[..., -1:]
        else:
            out_continuous = None
            out_last_continuous = None
        return out_discrete, out_continuous, new_mems, out_last_discrete, out_last_continuous

    def cal_loss(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            dropatt=args.dropout, tie_weight=True, 
                            d_embed=d_embed, div_val=div_val, 
                            tie_projs=tie_projs, pre_lnorm=True,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, 
                            cutoffs=cutoffs, attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
