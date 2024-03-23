# from gen.beatmap_generator import BeatmapGenerator
import math
import os
import random

import numpy as np
import torch
import torchaudio
import yaml
from einops import repeat
from torch import sqrt
from torch.special import expm1
from tqdm import tqdm
import matplotlib.pyplot as plt

from gen import gen_util, label_interpreter
from gen.interpreter import type_coord
from nn.net.transformer_xl.mem_transformer import MemTransformerLM
from util import audio_util, beatmap_util
from util.audio_util import cal_mel_spec
from util.general_util import recursive_to_cpu, recursive_to_ndarray, recursive_flatten_batch
from util.plt_util import plot_signal, plot_multiple_signal


def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * torch.log((torch.tan(t_min + t * (t_max - t_min))).clamp(min=1e-20))


class Generator:
    def __init__(self, device='cuda:0'):
        self.output_device = device

        self.mel_args = {
            'sample_rate': 22000,
            'n_fft': 400,
            'n_mels': 40,
        }
        self.n_beats = 16
        self.beat_divisor = 8
        self.mel_frame_per_snap = 8
        self.n_snaps = self.n_beats * self.beat_divisor
        self.subseq_mel_frames = self.n_snaps * self.mel_frame_per_snap
        self.batch_size = 16
        self.sample_rate = 22000
        self.n_fft = 400
        self.n_mels = 40
        self.n_metas = 3

        model_arg = {
            'n_token': 4,
            'n_layer': 8,
            'n_head': 8,
            'd_model': 256,
            'd_head': 32,
            'd_inner': 256,
            'dropout': 0.1,
            'dropatt': 0,
            'tie_weight': True,
            # same as d_model
            'd_embed': None,
            'div_val': 1,
            'tie_projs': [False],
            'pre_lnorm': False,
            'tgt_len': self.n_snaps,
            'ext_len': 0,
            'mem_len': self.n_snaps,
            'cutoffs': [],
            'adapt_inp': False,
            'same_length': False,
            'attn_type': 0,
            'clamp_len': -1,
            'sample_softmax': -1,
            'context_compress': self.mel_frame_per_snap,
            'd_context': self.n_mels + self.n_metas,
            # do not do coord
            'd_continuous': 0,
        }

        self.model = MemTransformerLM(**model_arg)
        project_root = os.path.dirname(os.path.dirname(__file__))
        pretrained_model_root = os.path.join(project_root, 'resources', 'pretrained_models')
        model_path = os.path.join(pretrained_model_root, 'transformer_xl', '20240311_transformer_xl_only_ho_type_offset',
                                  r'model_0_epoch_60_batch_-1.pt')
        # model_path = r'./resources/pretrained_models/model_0_epoch_2_batch_-1.pt'
        state_dict = torch.load(model_path, map_location=self.output_device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(self.output_device)

        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     new_state_dict[k[7:]] = v
        # self.model.load_state_dict(new_state_dict)
        self.label_interpreter = label_interpreter.MultiLabelInterpreter()

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

        ms_per_beat = 60000 / bpm
        snap_ms = ms_per_beat / self.beat_divisor

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
        extra_snaps, num_step = total_snaps % self.n_snaps, total_snaps // self.n_snaps
        print(total_snaps, total_mel_frames)
        print(extra_snaps, num_step)
        if extra_snaps > 0:
            mel_spec = np.concatenate([mel_spec, np.zeros([(self.n_snaps - extra_snaps) * self.mel_frame_per_snap, self.n_mels])], axis=0)
            num_step += 1
        mel_spec = mel_spec.reshape([num_step, self.subseq_mel_frames, self.n_mels]).transpose([2, 1, 0])[np.newaxis, ...]
        # 1, C, L, n_step
        print('mel spec shape: ', mel_spec.shape)

        offset = np.arange(num_step * self.subseq_mel_frames, dtype=np.float32) / self.subseq_mel_frames
        offset = np.transpose(offset.reshape([num_step, self.subseq_mel_frames]), [1, 0])
        offset = offset[np.newaxis, np.newaxis, ...]

        # mel_spec = torch.tensor(mel_spec, dtype=torch.float32, device=self.output_device)

        # run for every meta
        for index, meta_dict in enumerate(meta_list):
            star = meta_dict.pop('star')
            meta = np.array([
                star / 10, (bpm-50) / 360
            ], dtype=np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
            meta = np.tile(meta, [1, 1, self.subseq_mel_frames, num_step])

            # 1, C, L, n_step
            all_step_context = np.concatenate([mel_spec, offset, meta], axis=1)
            all_step_context = torch.from_numpy(all_step_context).float().to(self.output_device)

            all_step_out_discrete = []
            # all_step_out_continuous = []
            out_last_discrete = self.sample_start_token()
            rand_inst = random.Random()
            mems = []
            for step_idx in tqdm(range(num_step), total=num_step):
                context = all_step_context[..., step_idx]
                out_discrete, out_continuous, mems, out_last_discrete, out_last_continuous = self.model.forward_segment(
                    out_last_discrete,
                    None,
                    # out_last_continuous if self.do_coord else None,
                    context,
                    mems,
                    rand_inst=rand_inst,
                )

                all_step_out_discrete.append(out_discrete)
                # all_step_out_continuous.append(out_continuous)
            # N, L -> L,
            all_step_out_discrete = torch.cat(all_step_out_discrete, dim=1).detach().cpu().numpy()[0]
            # if self.do_coord:
            #     all_step_out_continuous = torch.cat(all_step_out_continuous, dim=1)
            # flatten steps
            # all_step_inp_discrete = all_step_inp_discrete.reshape(list(all_step_inp_discrete.shape[:-2]) + [-1])
            # all_step_inp_continuous = all_step_inp_continuous.reshape(list(all_step_inp_continuous.shape[:-2]) + [-1])
            # self.log_gen_output(all_step_out_discrete, all_step_out_continuous, epoch, batch_idx, 'output_')
            self.log_gen_output(
                all_step_out_discrete,
                save_path=os.path.join(os.path.dirname(osz_path),
                                       meta_dict['title'] + '_' + meta_dict['version'] + '.jpg')
            )
            # self.log_gen_output(all_step_inp_discrete, all_step_inp_continuous, epoch, batch_idx, 'input_')

            beatmap = beatmap_util.get_empty_beatmap()
            beatmap_util.set_bpm(beatmap, bpm, self.beat_divisor)
            beatmap_util.set_meta(
                beatmap,
                meta_dict
            )
            self.label_interpreter.gen_hitobjects(
                beatmap, all_step_out_discrete, start_time * 1000, snap_ms, self.beat_divisor
            )

            beatmap_name = beatmap_util.osu_filename(beatmap)
            osu_path = os.path.join(osu_dir, beatmap_name)
            beatmap.write_path(osu_path)

            osu_path_list.append(osu_path)
            beatmap_list.append(beatmap)
        beatmap_util.pack_to_osz(audio_file_path, osu_path_list, osz_path, beatmap_list)
        print('saved to %s' % osz_path)

    def sample_start_token(self):
        """
        N, 1
        """
        return torch.tensor([[1]], dtype=torch.long, device=self.output_device)

    def log_gen_output(self, out_discrete, save_path=None):
        # L, -> 4, L
        ho_type_one_hot = np.eye(4)[out_discrete].T
        ho_type_one_hot = ho_type_one_hot[1:]
        # circle
        ho_type_one_hot[0, ...] *= 3
        # slider
        ho_type_one_hot[1, ...] *= 2
        plot_multiple_signal(ho_type_one_hot,
                             'ho',
                             save_path,
                             show=True)


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
