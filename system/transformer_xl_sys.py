import collections
import math
import os
from functools import wraps

import numpy as np
import torch
from einops import repeat, reduce
from torch import sqrt
from torch import nn
from torch.nn import functional as F
from torch.special import expm1
from tqdm import tqdm

from system.base_sys import Train
from util.general_util import recursive_wrap_data
from util.plt_util import plot_signal


def init_weight(weight, init='normal',
                init_range=0.1,
                init_std=0.02, ):
    if init == 'uniform':
        nn.init.uniform_(weight, -init_range, init_range)
    elif init == 'normal':
        nn.init.normal_(weight, 0.0, init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m, proj_init_std=0.01,
                 init_std=0.02, ):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


class TransformerXLSYS(Train):
    def __init__(self,
                 config_dict,
                 task_type='classification',
                 **kwargs):
        super().__init__(config_dict, task_type, **kwargs)

        # training objective
        self.pred_objective = self.config_dict['pred_objective']
        assert self.pred_objective in {'v',
                                       'eps'}, 'whether to predict v-space (progressive distillation paper) or noise'

        noise_d = self.config_dict['noise_d']
        noise_d_low = self.config_dict['noise_d_low']
        noise_d_high = self.config_dict['noise_d_high']
        clip_sample_denoised = True
        min_snr_loss_weight = True
        min_snr_gamma = 5

        # image dimensions
        self.channels = self.config_dict['channels']
        self.length = self.config_dict['length']

        # noise schedule
        assert not all([*map(exists, (noise_d, noise_d_low,
                                      noise_d_high))]), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'

        # determine shifting or interpolated schedules

        self.log_snr = logsnr_schedule_cosine

        if exists(noise_d):
            # if noise_d == image_size, this is the same as logsnr_schedule_cosine
            self.log_snr = logsnr_schedule_shifted(self.log_snr, self.length, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), 'both noise_d_low and noise_d_high must be set'

            self.log_snr = logsnr_schedule_interpolated(self.log_snr, self.length, noise_d_low, noise_d_high)

        # sampling

        self.num_sample_steps = self.config_dict['num_sample_steps']
        self.clip_sample_denoised = clip_sample_denoised

        # loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        # if 'embedding_decoder' in self.config_dict:
        #     self.embedding_output_decoder = dynamic_import(self.config_dict['embedding_decoder'])(**self.config_dict['embedding_decoder_])

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next, cond_data):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.model(x, batch_log_snr, cond_data)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next, cond_data):
        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next, cond_data=cond_data)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_data):
        img = torch.randn(shape, device=self.output_device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.output_device)

        # for i in range(self.num_sample_steps):
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps,
                      ncols=80):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next, cond_data)

        return img

    @torch.no_grad()
    def sample(self, cond_data, batch_size=16):
        return self.p_sample_loop((batch_size, self.channels, self.length), cond_data)

    # training related functions - noise prediction

    # @autocast(enabled=False)
    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)
        # print('times', times)
        # print('log_snr', log_snr)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        # by property of sigmoid, alpha**2 + sigma**2 = 1
        x_noised = x_start * alpha + noise * sigma

        return x_noised, log_snr

    def p_losses(self, x_start, cond_data, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # times: 0~1 uniform
        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr, cond_data)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == 'eps':
            target = noise

        loss = F.mse_loss(model_out, target, reduction='none')

        loss = reduce(loss, 'b ... -> b', 'mean')

        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max=self.min_snr_gamma)

        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)

        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr
        else:
            raise ValueError

        return (loss * loss_weight).mean()

    def log_gen_output(self, gen_output, epoch, batch_idx, prefix='gen', plot_first=3):
        output_dir = os.path.join(self.log_dir, 'output', 'e%d_b%d' % (epoch, batch_idx), prefix)
        os.makedirs(output_dir, exist_ok=True)

        gen_output = gen_output.cpu().detach().numpy()
        coord_output, embedding_output = gen_output[:, :2], gen_output[:, 2:]

        # embedding_output = np.round(embedding_output)
        # -> [batch_size, n_snaps]
        all_sample_decoded = np.sum(np.round(embedding_output) * np.arange(1, 4).reshape([1, -1, 1]), axis=1)

        with open(os.path.join(output_dir, r'hit_signal.txt'), 'w') as f:
            for sample_decoded in all_sample_decoded:
                f.write(str(sample_decoded.tolist()) + '\n')
        with open(os.path.join(output_dir, r'cursor_signal.txt'), 'w') as f:
            for sample_coord_output in coord_output:
                # [n_snaps, coord_num]
                f.write(str(sample_coord_output.T.tolist()) + '\n')
        for i, (sample_coord_output, sample_embedding_output) in enumerate(zip(coord_output, embedding_output)):
            if i >= plot_first:
                break
            for signal, name in zip(
                    sample_coord_output,
                    [
                        'cursor_x', 'cursor_y'
                    ]
            ):
                fig_array, fig = plot_signal(signal, name,
                                             save_path=os.path.join(output_dir, name + '%d.jpg' % i),
                                             show=False)
                self.writer.add_figure(prefix + name, fig, epoch)
            # plot as float
            for signal, name in zip(
                    sample_embedding_output,
                    [
                        'circle_hit', 'slider_hit', 'spinner_hit'
                    ]
            ):
                fig_array, fig = plot_signal(signal, name,
                                             save_path=os.path.join(output_dir, name + '%d.jpg' % i),
                                             show=False)
                self.writer.add_figure(prefix + name, fig, epoch)

    def run_train(self):
        self.init_train_state()
        for epoch in range(self.start_epoch, self.epoch):
            epoch_loss_list = []

            bar = tqdm(enumerate(self.train_iter), total=len(self.train_iter), ncols=80)
            for batch_idx, batch in bar:
                batch = recursive_wrap_data(batch, self.output_device)
                loss_value = self.training_step(batch, batch_idx, epoch)
                bar.set_postfix(collections.OrderedDict({'loss': loss_value}))
                epoch_loss_list.append(loss_value)

            self.scheduler.step()
            self.logger.info("loss %.4f", sum(epoch_loss_list) / len(epoch_loss_list))

            self.save_model(epoch)
            self.save_train_state(self.model, self.optimizer, epoch)

            if self.log_epoch_step is not None and epoch % self.log_epoch_step == 0:
                # if self.current_epoch % 16 == 0:
                # log sampled images
                audio, x_start, meta = batch
                num_gen_vis = 3
                # log sampled images
                sample_imgs = self.sample(
                    (torch.cat([audio[:num_gen_vis], audio[:num_gen_vis]], dim=0),
                     [torch.cat([m[:num_gen_vis], m[:num_gen_vis]], dim=0) for m in meta]),
                    batch_size=num_gen_vis * 2
                )
                self.log_gen_output(sample_imgs[:num_gen_vis], epoch, -1, 'generated_seed1')
                self.log_gen_output(sample_imgs[num_gen_vis:], epoch, -1, 'generated_seed2')
                self.log_gen_output(x_start[:num_gen_vis], epoch, -1, 'input')
        self.save_model(-1)
        self.save_train_state(self.model, self.optimizer, -1)

    def training_step(self, batch, batch_idx, epoch):
        audio, x_start, meta = batch
        # print(meta)
        cond_data = (audio, meta)
        # input value -1~1
        times = torch.zeros((x_start.shape[0],), device=x_start.device).float().uniform_(0, 1)

        loss = self.p_losses(x_start, cond_data, times)

        if self.log_batch_step is not None and batch_idx % self.log_batch_step == 0:
            num_gen_vis = 3
            # log sampled images
            sample_imgs = self.sample(
                (torch.cat([audio[:num_gen_vis], audio[:num_gen_vis]], dim=0),
                 [torch.cat([m[:num_gen_vis], m[:num_gen_vis]], dim=0) for m in meta]),
                batch_size=num_gen_vis * 2
            )
            self.log_gen_output(sample_imgs[:num_gen_vis], epoch, batch_idx, 'generated_seed1')
            self.log_gen_output(sample_imgs[num_gen_vis:], epoch, batch_idx, 'generated_seed2')
            self.log_gen_output(x_start[:num_gen_vis], epoch, batch_idx, 'input')

            self.save_model(epoch, batch_idx)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len


def train():
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    for batch, (data, target, seq_len) in enumerate(train_iter):
        model.zero_grad()
        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = para_model(data_i, target_i, *mems[i])
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                train_loss += loss.float().item()
        else:
            ret = para_model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            train_loss += loss.float().item()

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss = evaluate(va_iter)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
            else:
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
            logging(log_str)
            logging('-' * 100)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if args.sample_softmax > 0:
                    scheduler_sparse.step(val_loss)

            eval_start_time = time.time()

        if train_step == args.max_step:
            break