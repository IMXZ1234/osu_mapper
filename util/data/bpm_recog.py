import numpy as np
import torchaudio
import torch
import matplotlib.pyplot as plt


def win_frame_sum(data, win_length, keep_dim=True):
    """
    sum up all frames in window
    frame: dim=2
    """
    ret_data = torch.zeros(data.shape)
    frame_num = data.shape[2]
    left = (win_length - 1) // 2
    right = win_length - left - 1
    # mirror padding
    ret_data[:, :, 0] = torch.sum(data[:, :, :left + 1], dim=2) * 2
    if right > left:
        ret_data[:, :, 0] += data[:, :, :right]
    for i in range(1, frame_num):
        frame_subtract = i - left
        if frame_subtract < 0:
            frame_subtract = -frame_subtract
        frame_add = i + right + 1
        if frame_add >= frame_num:
            frame_add = frame_num * 2 - frame_add - 2
        ret_data[:, :, i] = ret_data[:, :, i-1] - data[:, :, frame_subtract] + data[:, :, frame_add]
    return ret_data


def delete_continuous(list_in):
    if len(list_in) == 0:
        return []
    list_out = [list_in[0]]
    for i in range(1, len(list_in)):
        if list_in[i] != list_in[i-1] + 1:
            list_out.append(list_in[i])
    return list_out


if __name__ == '__main__':
    # temp_wav_path = audio_file_path[:-4] + '.wav'
    # audio_util.audio_to(audio_file_path, temp_wav_path)
    audio_path = r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\bgm\27_ライムライトの残火piano_version.ogg'
    # [channel, time]
    audio_data, sample_rate = torchaudio.backend.soundfile_backend.load(audio_path)
    audio_meta = torchaudio.backend.soundfile_backend.info(audio_path)
    print(audio_data.shape)
    tr = torchaudio.transforms.Resample(sample_rate, sample_rate / 2)
    resampled_audio_data = tr(audio_data)
    print(resampled_audio_data.shape)
    torch.stft()
    # print(audio_meta)
    # # [channel, freq, frame]
    win_length = 400
    ts = torchaudio.transforms.Spectrogram(n_fft=win_length)
    spectrogram = ts(audio_data)
    print(spectrogram.shape)
    plt.imshow(spectrogram[0][:, :1024], cmap='bwr')
    plt.colorbar()
    plt.show()
    resampled_spectrogram = ts(resampled_audio_data)
    print(resampled_spectrogram.shape)
    plt.imshow(resampled_spectrogram[0][:, :1024], cmap='bwr')
    plt.colorbar()
    plt.show()
    # print(1 / sample_rate * win_length)
    # print(sample_rate / win_length)
    # diff_spectrogram = torch.diff(spectrogram, dim=1)
    # diff_spectrogram[torch.where(diff_spectrogram < 0)] = 0
    # print(diff_spectrogram.shape)
    # cal_thresh_win_len = 11
    # diff_win_thresh_spectrogram = win_frame_sum(diff_spectrogram, win_length=cal_thresh_win_len) / cal_thresh_win_len * 1.5
    # sum_diff_spectrogram = torch.sum(torch.sum(diff_spectrogram, dim=0), dim=0)
    # print(sum_diff_spectrogram.shape)
    # sum_diff_win_thresh_spectrogram = torch.sum(torch.sum(diff_win_thresh_spectrogram, dim=0), dim=0)
    # print(sum_diff_win_thresh_spectrogram.shape)
    # beat_pos = torch.where(sum_diff_win_thresh_spectrogram < sum_diff_spectrogram)[0]
    # print(beat_pos)
    # print(len(beat_pos))
    # beat_pos = delete_continuous(beat_pos.data.tolist())
    # print(beat_pos)
    # print(len(beat_pos))
    # beat_interval = np.diff(beat_pos)
    # counts = [0 for _ in range(min(beat_interval), max(beat_interval) + 1)]
    # for itv in beat_interval:
    #     counts[itv - min(beat_interval)] += 1
    # print(counts)
    # overall_beat_interval = np.argmax(counts) + min(beat_interval)
    # print(overall_beat_interval)
    # bpm = 60 * win_length / 2 * overall_beat_interval / sample_rate
    # print(bpm)
    # plt.plot(list(range(len(sum_diff_spectrogram))), sum_diff_spectrogram)
    # plt.plot(list(range(len(sum_diff_win_thresh_spectrogram))), sum_diff_win_thresh_spectrogram)
    # plt.scatter(beat_pos, sum_diff_spectrogram[beat_pos], color='green')
    # plt.show()
    # plt.bar(list(range(min(beat_interval), max(beat_interval) + 1)), counts)
    # plt.show()
    # # win_length
