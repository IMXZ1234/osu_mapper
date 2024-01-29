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
import matplotlib.pyplot as plt

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
            'num_meta': 3,
            'audio_in_channels': 40,
            'audio_out_channels': 12,
            'audio_length_compress': 8,
        }

        self.mel_args = {
            'sample_rate': 22000,
            'n_fft': 400,
            'n_mels': 40,
        }
        self.n_beats = 64
        self.beat_divisor = 8
        self.mel_frame_per_snap = 8
        self.n_snaps = self.n_beats * self.beat_divisor
        self.n_frames = self.n_snaps * self.mel_frame_per_snap

        self.half_n_snaps = self.n_snaps // 2
        self.half_n_frames = self.n_frames // 2

        self.batch_size = 16
        self.sample_rate = 22000
        self.n_fft = 400
        self.n_mels = 40

        self.model = CUViT(**model_arg)
        project_root = os.path.dirname(os.path.dirname(__file__))
        pretrained_model_root = os.path.join(project_root, 'resources', 'pretrained_models')
        model_path = os.path.join(pretrained_model_root, 'cddpm',  r'model_0_epoch_150_batch_-1.pt')
        # model_path = r'./resources/pretrained_models/model_0_epoch_2_batch_-1.pt'
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
            audio_info_path = os.path.join(os.path.dirname(audio_file_path), audio_filename_no_ext + '.yaml')

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
            print('frame_length', frame_length)
            mel_spec = cal_mel_spec(
                audio_for_mel,
                frame_start, frame_length,
                window='hamming',
                nfft=self.n_fft,
                n_mel=self.n_mels,
                sample_rate=self.sample_rate
            )
            # -> n_frames, n_mels
            mel_spec = np.mean(mel_spec, axis=1)
            # print(mel_spec)
            assert mel_spec.shape[0] == total_mel_frames, 'mel_spec.shape[0] %s != total_mel_frames %s' % (mel_spec.shape[0], total_mel_frames)
            print('mel spec shape: ', mel_spec.shape)
            mel_spec -= (np.mean(mel_spec, axis=0) + 1e-8)
            print(mel_spec[:, 5])
            print(mel_spec)
            plt.figure()
            plt.title('mel_spec')
            plt.imshow(mel_spec.T, aspect='auto')
            plt.show()
        except Exception:
            print('mel spec generation failed!')
            return

        # with open(r'C:\Users\asus\coding\python\osu_mapper\resources\data\processed_v4\mel\999834.pkl', 'rb') as f:
        #     mel_spec = pickle.load(f)
        total_mel_frames = len(mel_spec)
        total_snaps = total_mel_frames // self.mel_frame_per_snap

        osu_path_list = []
        beatmap_list = []

        # pad tail to multiples of subseq length
        extra_snaps, num_subseq = total_snaps % self.half_n_snaps, total_snaps // self.half_n_snaps
        print(total_snaps, total_mel_frames)
        print(extra_snaps, num_subseq)
        if extra_snaps > 0:
            mel_spec = np.concatenate([mel_spec, np.zeros([(self.half_n_snaps - extra_snaps) * self.mel_frame_per_snap, self.n_mels])], axis=0)
            num_subseq += 1
        # [num_subseq * self.half_n_frames, self.n_mels]
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32, device=self.output_device)
        mel_spec = mel_spec.reshape([num_subseq, self.half_n_frames, self.n_mels])
        former_half, latter_half = mel_spec[:-1], mel_spec[1:]
        # -> [num_subseq-1, self.n_frames, self.n_mels]
        mel_spec = torch.cat([former_half, latter_half], dim=1)
        # -> N, C, L
        mel_spec = mel_spec.permute(0, 2, 1)

        print('mel spec shape: ', mel_spec.shape)
        offset_proportion = torch.arange(mel_spec.shape[0], dtype=torch.float32, device=self.output_device) * (self.half_n_snaps / total_snaps)

        # run for every meta
        for index, meta_dict in enumerate(meta_list):
            # get and remove 'star'
            # np.array([star - 3.5]) / 5
            star_meta = np.array([meta_dict.pop('star')], dtype=np.float32) / 10
            star_meta = torch.tensor(star_meta, device=self.output_device).expand([mel_spec.shape[0], -1])
            # print(bpm)
            bpm_meta = torch.tensor([(bpm-50) / 360], device=self.output_device).expand([mel_spec.shape[0], -1])
            # print(star_meta, bpm_meta)
            gen_output = self.sample_autoreg((mel_spec, [star_meta, offset_proportion, bpm_meta]))
            # self.log_gen_output(gen_output.permute(1, 0, 2).reshape([1, 5, -1]), -1)

            gen_output = recursive_to_cpu(gen_output)
            gen_output = recursive_to_ndarray(gen_output)
            # N, 5, L -> N, L, 5
            gen_output = gen_output.transpose(0, 2, 1).reshape([-1, 5])
            # gen_output = recursive_flatten_batch(gen_output, cat_dim=2).T

            self.log_gen_output(gen_output, save_dir=None, index=index)

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

    @torch.no_grad()
    def p_sample(self, x, time, time_next, cond_data):

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

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(posterior_variance) * noise

    @torch.no_grad()
    def p_sample_loop_autoreg(self, shape, cond_data):
        """
        shape: num_segment, channel, segment_length

        for every time step, the former half of diffusion input is replaced with latter half of the output
        of the last segment from the former time step

        this may improve local coherence
        """
        N, C, L = shape
        x = torch.randn(shape, device=self.output_device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.output_device)

        # for i in range(self.num_sample_steps):
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps, ncols=80):
            times = steps[i]
            times_next = steps[i + 1]
            x = self.p_sample(x, times, times_next, cond_data)
            x[1:, ..., :L//2] = x[:N-1, ..., L//2:]

        return x

    @torch.no_grad()
    def sample_autoreg(self, cond_data):
        return self.p_sample_loop_autoreg((cond_data[0].shape[0], self.channels, self.length), cond_data)

    def log_gen_output(self, gen_output, save_dir=None, index=0):
        if save_dir is None:
            save_dir = r'./resources/generate/vis'
        os.makedirs(save_dir, exist_ok=True)
        gen_output = gen_output.T
        coord_output, embedding_output = gen_output[:2], gen_output[2:]

        fig = plt.figure()
        for signal, name in zip(
            coord_output,
            [
                'cursor_x', 'cursor_y'
            ]
        ):
            plt.plot(signal)
            plt.savefig(os.path.join(save_dir, 'cursor%d.png' % index))
        # plot as float
        fig = plt.figure()
        for signal, name in zip(
            embedding_output,
            [
                'circle_hit', 'slider_hit', 'spinner_hit'
            ]
        ):
            plt.plot(signal)
            plt.savefig(os.path.join(save_dir, 'type%d.png' % index))
        # plt.show()



if __name__ == '__main__':
    import slider
    import sys
    import pickle
    import torchaudio
    # root_dir = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\audio'
    # audio_path = os.path.join(root_dir, 'audio_fix.mp3')
    # audio, sr = audio_util.audioread_get_audio_data(audio_path)
    # torchaudio.functional.resample(audio, sr, 16000)
    # mel = torchaudio.transforms.MelSpectrogram(n_mels=40)
    # mel_spec = mel(audio).numpy()
    # print(mel_spec.shape)
    # mel_spec = mel_spec.mean(axis=0)
    # print(np.min(mel_spec), np.max(mel_spec), np.mean(mel_spec))
    # mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)  # Numerical Stability
    # mel_spec = 20 * np.log10(mel_spec)  # dB
    # mel_spec -= (np.mean(mel_spec, axis=0) + 1e-8)
    # print(np.min(mel_spec), np.max(mel_spec), np.mean(mel_spec))
    #
    # plt.figure()
    # plt.title('from dataset mel spec torchaudio')
    # plt.imshow(mel_spec, aspect='auto')
    # plt.show()
    #
    # plt.figure()
    # plt.hist(mel_spec.reshape([-1]), bins=100)
    # plt.title('mel spec hist torchaudio')
    # plt.show()
    #
    # sys.exit()
    # bmp = slider.Beatmap.from_path(
    #     os.path.join(root_dir, r'CHiCO with HoneyWorks - Ai no Scenario (TV Size) (rew0825) [Cup].osu')
    # )
    # print(bmp.bpm_min(), bmp.bpm_max())

    root_dir = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\preprocessed_v7\mel'
    with open(
        os.path.join(root_dir, '999834.pkl'), 'rb'
    ) as f:
        mel_spec = pickle.load(f)
    print('from dataset mel spec', mel_spec.shape)
    print(mel_spec)
    print(np.min(mel_spec), np.max(mel_spec), np.mean(mel_spec))
    mel_spec -= (np.mean(mel_spec, axis=0) + 1e-8)
    print(np.min(mel_spec), np.max(mel_spec), np.mean(mel_spec))

    plt.figure()
    plt.hist(mel_spec.reshape([-1]), bins=100)
    plt.title('mel spec hist')
    plt.show()

    plt.figure()
    plt.title('from dataset mel spec')
    plt.imshow(mel_spec.T, aspect='auto')
    plt.show()
    # sys.exit()

    # generator = Generator()
    # generator.generate_beatmapset(
    #     osz_path=os.path.join(root_dir, 'test.osz'),
    #     audio_file_path=os.path.join(root_dir, 'audio_fix.mp3'),
    #     meta_list=[
    #         {
    #             'audio_filename': 'audio_fix.mp3',  # indispensable
    #             'artist_unicode': 'ai',
    #             'artist': 'ai',  # indispensable
    #             'title_unicode': 'ai',
    #             'title': 'ai',  # indispensable
    #             'version': '8',  # indispensable
    #             'creator': 'IMXZ123',
    #             'circle_size': 3,
    #             'approach_rate': 8,
    #             'slider_tick_rate': 2,
    #             'star': 4.5,
    #         },
    #     ],
    #     audio_info_path=os.path.join(root_dir, 'audio_fix.yaml'),
    # )
