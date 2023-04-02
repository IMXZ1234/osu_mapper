import os

import audioread
import numpy as np
import torch
import torchaudio

FFMPEG_PATH = r'C:\Users\asus\coding\vsc++\ffmpeg-4.4.1-full_build-shared\bin\ffmpeg.exe'
FFPROBE_PATH = r'C:\Users\asus\coding\vsc++\ffmpeg-4.4.1-full_build-shared\bin\ffprobe.exe'
OUTFILE_PATH = r'C:\Users\asus\coding\vsc++\ffmpeg-4.4.1-full_build-shared\bin\temp.txt'


class FFProbeMeta:
    def __init__(self):
        self.meta_dict = {}
        self.duration = 0
        self.start = 0
        self.bitrate = 0
        self.stream_meta_dict_list = []
        # list of audio info: [encoding, sample_rate, stereo, fltp, bitrate]
        self.stream_audio_list = []

    @staticmethod
    def _parse_meta(lines, i):
        meta_dict = {}
        leading_space_char_num = len(lines[i]) - len(lines[i].lstrip())
        while True:
            k, v = lines[i].split(':')
            meta_dict[k.strip()] = v.strip()
            i += 1
            if i >= len(lines) or leading_space_char_num != len(lines[i]) - len(lines[i].lstrip()):
                return meta_dict, i

    def _parse(self, ffprobe_dump_path):
        with open(ffprobe_dump_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('Input #'):
                # parse input info
                i += 1
                while i < len(lines) and (not lines[i].startswith('Input #')):
                    # print(i)
                    if i < len(lines) and lines[i].startswith('  Metadata:'):
                        i += 1
                        self.meta_dict, i = FFProbeMeta._parse_meta(lines, i)
                    if i < len(lines) and lines[i].startswith(r'  Duration:'):
                        properties = lines[i].split(',')
                        time_parts = properties[0].split(':')[1:]
                        self.duration = ((float(time_parts[0]) * 60) + float(time_parts[1]) * 60) + float(time_parts[2])
                        self.start = float(properties[1].split(':')[1])
                        self.bitrate = int()
                        i += 1
                    if i < len(lines) and lines[i].startswith(r'  Stream #'):
                        # parse stream info
                        s = lines[i].split(':')
                        parts = s[-1].split(',')
                        parts = [part.strip() for part in parts]
                        # sample_rate
                        parts[1] = int(parts[1].split(' ')[0])
                        # bitrate
                        parts[4] = int(parts[4].split(' ')[0])
                        self.stream_audio_list.append(parts)
                        i += 1
                        while i < len(lines) and (not lines[i].startswith('  Stream #')):
                            if i < len(lines) and lines[i].startswith('    Metadata:'):
                                i += 1
                                meta_dict, i = FFProbeMeta._parse_meta(lines, i)
                                self.stream_meta_dict_list.append(meta_dict)
                            else:
                                i += 1
            else:
                i += 1

    @classmethod
    def from_path(cls, audio_file_path):
        out_file_path = audio_file_path[:-4] + '_temp'
        while os.path.exists(out_file_path):
            out_file_path += '%'
        os.system(FFPROBE_PATH + ' \"%s\" > \"%s\" 2>&1' % (audio_file_path, out_file_path))
        inst = cls()
        inst._parse(out_file_path)
        os.remove(out_file_path)
        return inst

    def get_sample_rate(self, stream=0):
        assert stream < len(self.stream_audio_list)
        return self.stream_audio_list[stream][1]


def get_sample_rate_from_ffprobe_dump(ffprobe_dump_path):
    sample_rate = -1
    with open(ffprobe_dump_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('  Stream #0:0: Audio: '):
                return int(line.split(',')[1].strip().split(' ')[0])
    return sample_rate


def audio_convert(from_path, to_path):
    # print('converting %s to %s' % (from_path, to_path)) > %s 2>&1, OUTFILE_PATH >nul 2>nul
    os.system(FFMPEG_PATH + ' -i \"%s\" \"%s\" -y >nul 2>nul' % (from_path, to_path))


def dump_audio_info(audio_file_path, ffprobe_dump_path):
    os.system(FFPROBE_PATH + ' \"%s\" > \"%s\" 2>&1' % (audio_file_path, ffprobe_dump_path))


def get_audio_attr(audio_file_path, attr):
    # make sure existent file is not overwritten by temp gen
    out_file_path = audio_file_path[:-4] + '_temp'
    while os.path.exists(out_file_path):
        out_file_path += '%'
    os.system(FFPROBE_PATH + ' \"%s\" > \"%s\" 2>&1' % (audio_file_path, out_file_path))
    value = None
    if attr == 'sample_rate':
        value = get_sample_rate_from_ffprobe_dump(out_file_path)
    else:
        meta = FFProbeMeta.from_path(out_file_path)
        if attr in meta.meta_dict:
            value = meta.meta_dict[attr]
        else:
            try:
                value = getattr(meta, attr)
            except AttributeError:
                print('Unknown attribute for FFProbeMeta')
    os.remove(out_file_path)
    return value


def audioread_get_audio_data(audio_file_path, ret_tensor=True):
    """
    Conforms to torchaudio I/O api gen format.

    If ret_tensor return pytorch tensor, else return ndarray.
    """
    if isinstance(audio_file_path, str):
        f = audioread.audio_open(audio_file_path)
    else:
        f = audio_file_path
        # channel, sample_rate, duration can be obtained by f.channels, f.samplerate, f.duration
    audio_cs = []
    buf_num = 0
    total_len = 0
    for buf_i, buf in enumerate(f):
        data = np.frombuffer(buf, np.int16)
        audio_cs.append(np.stack([data[ci::f.channels] for ci in range(f.channels)], axis=0))
        total_len += len(data)
        buf_num += 1
    # normalize to (-1, 1)
    audio_cs = np.concatenate(audio_cs, axis=1, dtype=float) / (2 ** 15)
    if isinstance(audio_file_path, str):
        f.close()
    if ret_tensor:
        return torch.tensor(audio_cs, dtype=torch.float), f.samplerate
    return audio_cs, f.samplerate


def save_audio(file_path, audio_data, sample_rate, temp_wav_file_path=None):
    """
    audio_data should be tensor
    """
    name, ext = os.path.splitext(file_path)
    if ext in ('.wav', ".ogg", ".vorbis", ".flac", ".sph"):
        torchaudio.backend.soundfile_backend.save(file_path, audio_data, sample_rate)
        return
    else:
        if temp_wav_file_path is None:
            temp_wav_file_path = name + '.wav'
        torchaudio.backend.soundfile_backend.save(temp_wav_file_path, audio_data, sample_rate)
        audio_convert(temp_wav_file_path, file_path)
        os.remove(temp_wav_file_path)


def crop_audio(audio_file_path: str, save_audio_file_path: str = None,
               frame_num: int = 16384 * 48, offset: int = 0):
    if save_audio_file_path is None:
        save_audio_file_path = os.path.join(
            os.path.dirname(audio_file_path),
            'cropped_%d_' % frame_num + os.path.basename(audio_file_path)
        )
    audio_data, sample_rate = audioread_get_audio_data(audio_file_path)
    cropped_audio_data = audio_data[:, offset:offset + frame_num]
    save_audio(save_audio_file_path, cropped_audio_data, sample_rate)


def audio_len(audio_file_path: str):
    audio_data, sample_rate = audioread_get_audio_data(audio_file_path)
    return audio_data.shape[1] / sample_rate


if __name__ == '__main__':
    crop_audio(
        r'C:\Users\asus\coding\python\osu_mapper\resources\data\audio\47065.mp3',

    )
