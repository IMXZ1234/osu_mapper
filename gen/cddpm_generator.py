# from gen.beatmap_generator import BeatmapGenerator
import math
import os

import numpy as np
import torch
import torchaudio
import yaml
from einops import repeat
from torch import sqrt
from torch.special import expm1
from tqdm import tqdm

from gen import gen_util
from gen.interpreter import type_coord
from nn.net.diffusion.cuvit import CUViT
from util import audio_util, beatmap_util
from util.audio_util import cal_mel_spec
from util.general_util import recursive_to_cpu, recursive_to_ndarray, recursive_flatten_batch
from util.plt_util import plot_signal


def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * torch.log((torch.tan(t_min + t * (t_max - t_min))).clamp(min=1e-20))


class Generator:
    def __init__(self, device='cuda:0'):
        self.output_device = device

        model_arg = {
            'dim': 48,
            'channels': 5,
            'num_meta': 2,
            'audio_in_channels': 40,
            'audio_out_channels': 12,
        }

        self.mel_args = {
            'sample_rate': 22000,
            'n_fft': 512,
            'hop_length': 220,
            'n_mels': 40,
        }
        self.n_beats = 32
        self.beat_divisor = 8
        self.n_snaps = self.n_beats * self.beat_divisor
        self.batch_size = 16
        self.sample_rate = 22000
        self.mel_frame_per_snap = 16
        self.n_mels = 40

        self.model = CUViT(**model_arg)
        model_path = r'./resources/pretrained_models/model_0_epoch_2_batch_-1.pt'
        state_dict = torch.load(model_path, map_location=self.output_device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(self.output_device)

        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     new_state_dict[k[7:]] = v
        # self.model.load_state_dict(new_state_dict)
        self.label_interpreter = type_coord.TypeCoordInterpreter()

        # self.embedding_output_decoder = embedding_decode.EmbeddingDecode(
        #     r'./resources/pretrained_models/embedding_center-1_-1_normed.pkl',
        #     self.beat_divisor,
        #     r'./resources/pretrained_models/counter.pkl',
        # )
        self.num_sample_steps = 500
        self.channels = 5
        self.length = self.n_snaps
        self.log_snr = logsnr_schedule_cosine
        self.pred_objective = 'eps'

    @staticmethod
    def get_bpm_start_end_time(audio_file_path, audio_info_path):
        if not os.path.exists(audio_info_path):
            print('extracting bpm, start_time, end_time...')
            audio_info = gen_util.extract_bpm(audio_file_path)
            audio_info_dict = {
                'bpm': audio_info[0],
                'start_time': audio_info[1],
                'end_time': audio_info[2],
            }
            print(audio_info_dict)
            with open(audio_info_path, 'wt') as f:
                yaml.dump(audio_info_dict, f)
            print('saved audio info to %s' % audio_info_path)
        else:
            with open(audio_info_path, 'rt') as f:
                audio_info_dict = yaml.load(f, Loader=yaml.FullLoader)
            print(audio_info_dict)
            audio_info = [
                audio_info_dict['bpm'],
                audio_info_dict['start_time'],
                audio_info_dict['end_time'],
            ]
        return audio_info

    def generate_beatmapset(self,
                            osz_path,
                            audio_file_path,
                            meta_list,
                            audio_info_path=None,
                            osu_dir=None,
                            **kwargs):
        print('generating for %s' % audio_file_path)
        audio_filename_no_ext = os.path.splitext(os.path.basename(audio_file_path))[0]
        if osu_dir is None:
            osu_dir = os.path.dirname(audio_file_path)
        os.makedirs(osu_dir, exist_ok=True)
        if audio_info_path is None:
            audio_info_path = os.path.join(os.path.dirname(audio_file_path), audio_filename_no_ext + 'yaml')

        try:
            audio_data, sr = audio_util.audioread_get_audio_data(audio_file_path, ret_tensor=True)
        except Exception:
            print('fail to load audio %s' % audio_file_path)
            return
        audio_data = torchaudio.functional.resample(audio_data, sr, self.sample_rate)
        audio_data = audio_data.numpy()

        bpm, start_time, end_time = self.get_bpm_start_end_time(audio_file_path, audio_info_path)
        # from ms to sec
        start_time, end_time = start_time / 1000, end_time / 1000

        beat_length_s = 60 / bpm
        snap_length_s = beat_length_s / self.beat_divisor
        snap_length_samples_f = snap_length_s * self.sample_rate
        beat_length_samples_f = beat_length_s * self.sample_rate
        hop_length_s = snap_length_s / self.mel_frame_per_snap
        hop_length_samples_f = hop_length_s * self.sample_rate
        hop_length = math.ceil(hop_length_samples_f)
        # since we align mel frames with snaps,
        # the center of the first mel frame should depend on temporal position of the first timing point
        first_tp_s = start_time
        first_tp_sample_f = first_tp_s * self.sample_rate

        total_sample = audio_data.shape[1]
        # count by beat_length_samples_f backward and forward to retrieve aligned audio
        beats_before_first_tp = math.floor(first_tp_sample_f / beat_length_samples_f)
        crop_start_sample_f = first_tp_sample_f - beats_before_first_tp * beat_length_samples_f
        snaps_before_first_tp = beats_before_first_tp * self.beat_divisor
        beats_after_first_tp = math.floor((total_sample - first_tp_sample_f) / beat_length_samples_f)
        snaps_after_first_tp = beats_after_first_tp * self.beat_divisor
        crop_end_sample_f = first_tp_sample_f + beats_after_first_tp * beat_length_samples_f
        # crop_start_sample, crop_end_sample = round(crop_start_sample_f), round(crop_end_sample_f)

        total_beats_f = (crop_end_sample_f - crop_start_sample_f) / beat_length_samples_f
        total_snaps_f = total_beats_f * self.beat_divisor
        total_snaps = round(total_snaps_f)

        total_mel_frames = total_snaps * self.mel_frame_per_snap

        """
        process mel
        """
        try:
            frame_length = round(2 * hop_length_samples_f)
            frame_start_f = np.linspace(crop_start_sample_f - hop_length_samples_f, crop_end_sample_f - hop_length_samples_f, total_mel_frames, endpoint=False)
            frame_start = np.round(frame_start_f).astype(int)
            audio_for_mel = audio_data
            # pad and crop left
            if frame_start[0] < 0:
                audio_for_mel = np.concatenate([np.zeros([2, -frame_start[0]]), audio_for_mel], axis=1)
            else:
                audio_for_mel = audio_for_mel[..., frame_start[0]:]
            frame_start -= frame_start[0]
            # pad right, generally this will not happen
            if frame_start[-1] + frame_length > audio_for_mel.shape[1]:
                audio_for_mel = np.concatenate([audio_for_mel, np.zeros([2, frame_start[-1] + frame_length - audio_for_mel.shape[1]])], axis=1)
            else:
                audio_for_mel = audio_for_mel[..., :frame_start[-1] + frame_length]
            mel_spec = cal_mel_spec(
                audio_for_mel,
                frame_start, frame_length,
                window='hamming',
                nfft=frame_length,
                n_mel=self.n_mels,
                sample_rate=self.sample_rate
            )
            # -> n_frames, n_mels
            mel_spec = np.mean(mel_spec, axis=1)
            # print(mel_spec)
            assert mel_spec.shape[0] == total_mel_frames, 'mel_spec.shape[0] %s != total_mel_frames %s' % (mel_spec.shape[0], total_mel_frames)
            print('mel spec shape: ', mel_spec.shape)
        except Exception:
            print('mel spec generation failed!')
            return

        # with open(r'C:\Users\asus\coding\python\osu_mapper\resources\data\processed_v4\mel\999834.pkl', 'rb') as f:
        #     mel_spec = pickle.load(f)
        total_mel_frames = len(mel_spec)
        total_snaps = total_mel_frames // 16

        osu_path_list = []
        beatmap_list = []

        # pad tail to multiples of subseq length
        extra_snaps, num_subseq = total_snaps % self.n_snaps, total_snaps // self.n_snaps
        print(total_snaps, total_mel_frames)
        print(extra_snaps, num_subseq)
        if extra_snaps > 0:
            mel_spec = np.concatenate([mel_spec, np.zeros([(self.n_snaps - extra_snaps) * self.mel_frame_per_snap, self.n_mels])], axis=0)
            num_subseq += 1
        mel_spec = mel_spec.T.reshape([num_subseq, self.n_mels, self.n_snaps * self.mel_frame_per_snap])
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32, device=self.output_device)
        print('mel spec shape: ', mel_spec.shape)
        offset_proportion = torch.arange(num_subseq, dtype=torch.float32, device=self.output_device) * (self.n_snaps / total_snaps)

        # run for every meta
        for meta_dict in meta_list:
            # get and remove 'star'
            # np.array([star - 3.5]) / 5
            star_meta = np.array([meta_dict.pop('star')], dtype=np.float32) / 10
            star_meta = torch.tensor(star_meta, device=self.output_device).expand([mel_spec.shape[0], -1])
            gen_output = self.sample((mel_spec, [star_meta, offset_proportion]), mel_spec.shape[0])
            # self.log_gen_output(gen_output, 3)

            gen_output = recursive_to_cpu(gen_output)
            gen_output = recursive_to_ndarray(gen_output)
            # N, 5, L -> N, L, 5
            gen_output = gen_output.transpose(0, 2, 1).reshape([-1, 5])
            # gen_output = recursive_flatten_batch(gen_output, cat_dim=2).T

            # plot_gen_output(gen_output)

            beatmap = beatmap_util.get_empty_beatmap()
            beatmap_util.set_bpm(beatmap, bpm, self.beat_divisor)
            beatmap_util.set_meta(
                beatmap,
                meta_dict
            )
            self.label_interpreter.gen_hitobjects(
                beatmap, gen_output, bpm, start_time * 1000, end_time * 1000, self.beat_divisor
            )

            beatmap_name = beatmap_util.osu_filename(beatmap)
            osu_path = os.path.join(osu_dir, beatmap_name)
            beatmap.write_path(osu_path)

            osu_path_list.append(osu_path)
            beatmap_list.append(beatmap)
        beatmap_util.pack_to_osz(audio_file_path, osu_path_list, osz_path, beatmap_list)
        print('saved to %s' % osz_path)

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
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps, ncols=80):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next, cond_data)

        return img

    @torch.no_grad()
    def sample(self, cond_data, batch_size=16):
        return self.p_sample_loop((batch_size, self.channels, self.length), cond_data)

    def log_gen_output(self, gen_output, plot_first=3):
        gen_output = gen_output.cpu().detach().numpy()
        coord_output, embedding_output = gen_output[:, :2], gen_output[:, 2:]

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
                                             save_path=None,
                                             show=True)
            # plot as float
            for signal, name in zip(
                sample_embedding_output,
                [
                    'circle_hit', 'slider_hit', 'spinner_hit'
                ]
            ):
                fig_array, fig = plot_signal(signal, name,
                                             save_path=None,
                                             show=True)
