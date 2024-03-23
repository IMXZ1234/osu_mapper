import itertools
import os
import pickle
import zipfile
from datetime import timedelta

import numpy as np
import slider
import torch
import torchaudio
from matplotlib import pyplot as plt
from tqdm import tqdm

from nn.dataset import dataset_util
from preprocess import db, filter
from util import beatmap_util, audio_util, plt_util
from collections import Counter

np.set_printoptions(threshold=np.inf)


def power_to_db(specgram):
    specgram = np.where(specgram == 0, np.finfo(float).eps, specgram)
    return 10 * np.log10(specgram)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    """
    specgram: channels, length, n_mels
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    # im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    im = axs.imshow(power_to_db(specgram[0]), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_wave(wave, title=None, ylabel="ampl"):
    """
    wave: channels, length, ampl
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "ampl")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    # im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    axs.plot(np.arange(len(wave[0])), wave[0])
    # fig.colorbar(im, ax=axs)
    plt.show(block=False)


def view_osu():
    osu_path = r"C:\Users\asus\AppData\Local\osu!\Songs\1013342 Sasaki Sayaka - Sakura, Reincarnation\Sasaki Sayaka - Sakura, Reincarnation (Riana) [Mimari's Hard].osu"
    beatmap = slider.Beatmap.from_path(osu_path)
    print(beatmap.beat_divisor)
    snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
    label = dataset_util.hitobjects_to_label_v2(beatmap, snap_ms=snap_ms, multi_label=True)
    print(label)
    print([type(ho) for ho in beatmap._hit_objects])
    for ho in beatmap._hit_objects:
        if isinstance(ho, slider.beatmap.Slider):
            print(ho.num_beats)
            print(ho.time)
            print(ho.end_time)
            print(ho.position)


def view_model():
    model_path = r'D:\osu_mapper\resources\result\seqganv2\0.01\1\model_0_epoch_-1.pt'
    model = torch.load(model_path, map_location='cpu')
    gru_ih_l0 = model['gru.weight_ih_l0']
    print(gru_ih_l0.shape)
    # 128 + 128 + 514
    print(gru_ih_l0)
    embed, pos_embed, feature = gru_ih_l0[:, :128], gru_ih_l0[:, 128:256], gru_ih_l0[:, 256:]
    for item in [embed, pos_embed, feature]:
        print(torch.norm(item))
        print(torch.sum(torch.abs(item)) / torch.numel(item))


def find_single_elem(label, elem=2):
    total_len = 0
    pos = 0
    num_single_elem = 0
    while pos < len(label):
        while label[pos] == elem:
            total_len += 1
            pos += 1
            if pos >= len(label):
                break
        if total_len == 1:
            num_single_elem += 1
        total_len = 0
        pos += 1
    return num_single_elem


def find_sequence_elem(label, elem=1):
    total_len = 0
    pos = 0
    num_seq_elem = 0
    while pos < len(label):
        while label[pos] == elem:
            total_len += 1
            pos += 1
            if pos >= len(label):
                break
        if total_len > 1:
            num_seq_elem += 1
        total_len = 0
        pos += 1
    return num_seq_elem


def view_dataset():
    with open(
        r'./resources/data/fit/label_pos/label.pkl',
        'rb'
    ) as f:
        label_list = pickle.load(f)
    print(len(label_list))
    print(label_list[0])

    with_1_seq = 0

    for label in label_list:
        type_label = label[:, 0]
        num = find_sequence_elem(type_label, 1)
        if num > 0:
            with_1_seq += 1
            print(num)
            print(type_label)
    print('with_1_seq')
    print(with_1_seq)
    # with open(
    #     r'C:\Users\asus\coding\python\osu_mapper\resources\data\fit\label_pos\data.pkl',
    #     'rb'
    # ) as f:
    #     data_list = pickle.load(f)
    # print(len(data_list))
    # # print(data_list[0])
    # print(data_list[0].shape)
    # data = data_list[0]
    # print(np.mean(data))
    # print(np.max(data))
    # print(np.min(data))
    # for label in label_list:
    #     pos_less_0 = np.where(label < 0)
    #     print(label[pos_less_0])
    # total_over_1 = 0
    # max_x = 0
    # max_y = 0
    # for label in label_list:
    #     max_x = max(max_x, np.max(label[:, 1]))
    #     max_y = max(max_y, np.max(label[:, 2]))
    # print(max_x)
    # print(max_y)
    # min_x = 1
    # min_y = 1
    # for label in label_list:
    #     min_x = min(min_x, np.min(label[:, 1]))
    #     min_y = min(min_y, np.min(label[:, 2]))
    # print(min_x)
    # print(min_y)
        # if (label[:, 1:] == 1.).any():
        #     where_1 = np.where(label[:, 1:] == 1.)
        #     print(where_1)
        #     where_1 = (where_1[0], where_1[1] + 1)
        #     print(label[where_1])
            # print(label)


def view_beatmap():
    beatmap = slider.Beatmap.from_path(
        # r'C:\Users\asus\coding\python\osu_mapper\resources\solfa feat. Ceui - primal (Shikibe Mayu) [Expert].osu'
        # r'C:\Users\asus\AppData\Local\osu!\Songs\1774219 EPICA - Wings of Freedom\EPICA - Wings of Freedom (Luscent) [Alis Libertatis].osu'
        r'C:\Users\asus\AppData\Local\osu!\Songs\beatmap-638052960603827278-audio\zato - jiyuu (IMXZ123) [Insane].osu'
    )
    snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
    tp_list = beatmap.timing_points
    tp_list = [tp for tp in tp_list if tp.parent is not None]
    # next_tp_idx = 0
    # next_tp = tp_list[0]
    # current_sv = -100
    # print(next_tp.ms_per_beat)
    for ho in beatmap.hit_objects():
        if isinstance(ho, slider.beatmap.Slider):
            # if ho.time > next_tp.offset:
            #     next_tp_idx += 1
            #     current_sv = next_tp.ms_per_beat
            length = int((ho.end_time - ho.time) / timedelta(milliseconds=1) / snap_ms)
            # # if length <= 4:
            # #     print(ho)
            # #     print(length)
            pos = beatmap_util.slider_snap_pos(ho, length)
            pos_at_ticks = ho.tick_points
            for p in pos:
                if p.x < 0 or p.y < 0:
                    print(pos)
                    print(pos_at_ticks)


def from_db_with_filter():
    table_name = 'MAIN'
    train_db = db.OsuDB(
        r'./resources/data/osu_train_mel.db'
    )
    ids = train_db.all_ids(table_name)
    sample_filter = filter.OsuTrainDataFilterGroup(
        [
            filter.ModeFilter(),
            filter.SnapDivisorFilter(),
            filter.BeatmapsetSingleAudioFilter(),
            filter.HitObjectFilter(),
            filter.SingleUninheritedTimingPointFilter(),
            filter.SingleBMPFilter(),
            filter.SnapInNicheFilter(),
        ]
    )
    for id_ in ids:
        record = train_db.get_record(id_, table_name)
        # first_ho_snap = record[db.OsuDB.EXTRA_START_POS + 1]
        beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
        audio_path = os.path.join(r'./resources/data/mel', record[db.OsuDB.AUDIOFILENAME_POS])
        if not sample_filter.filter(beatmap, audio_path):
            continue
        yield beatmap, audio_path


def view_mel():
    i = 5
    for beatmap, audio_path in from_db_with_filter():
        with open(audio_path, 'rb') as f:
            audio_mel = pickle.load(f)
        print(audio_mel.shape)
        plot_spectrogram(audio_mel[0])
        # plt.imshow(audio_mel[0][:, :1000])
        # plt.show()
        # print(audio_mel)
        i -= 1
        if i < 0:
            break


def view_distribution(list_in, title=None, save=True):
    list_in = np.array(list_in)
    min_value, mean_value, max_value = np.min(list_in), np.mean(list_in), np.max(list_in)
    std = np.std(list_in)
    plt.hist(list_in)
    if title is None:
        title = 'plot'
    plt.title(title)
    if save:
        plt.savefig(r'./resources/vis/%s_%.4f_%.4f_%.4f_%.4f.png' % (title, min_value, mean_value, max_value, std))
    plt.show()
    with open('./resources/vis/meta/%s.pkl' % title, 'wb') as f:
        pickle.dump(list_in, f)


def ho_density_distribution():
    ho_density_list = []
    blank_proportion_list = []
    num_circle_list, num_slider_list = [], []
    circle_proportion = []
    overall_difficulty_list = []
    approach_rate_list = []
    for beatmap, audio_path in from_db_with_filter():
        assert isinstance(beatmap, slider.Beatmap)
        snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
        label = dataset_util.hitobjects_to_label_v2(beatmap, snap_ms=snap_ms)
        # total_snap_num = beatmap_util.get_total_snaps(beatmap)
        total_snap_num = len(label)
        ho_density_list.append(len(beatmap._hit_objects) / total_snap_num)
        num_circle, num_slider = 0, 0
        for ho in beatmap._hit_objects:
            if isinstance(ho, slider.beatmap.Circle):
                num_circle += 1
            elif isinstance(ho, slider.beatmap.Slider):
                num_slider += 1
        num_circle_list.append(num_circle)
        num_slider_list.append(num_slider)
        circle_proportion.append(float(num_circle) / float(num_circle + num_slider))
        blank_proportion_list.append(len(np.where(np.array(label) == 0)[0]) / len(label))
        overall_difficulty_list.append(beatmap.overall_difficulty)
        approach_rate_list.append(beatmap.approach_rate)
    view_distribution(ho_density_list, 'ho_density')
    view_distribution(blank_proportion_list, 'blank_proportion')
    view_distribution(num_circle_list, 'num_circle')
    view_distribution(num_slider_list, 'num_slider')
    view_distribution(circle_proportion, 'circle_proportion')
    view_distribution(overall_difficulty_list, 'overall_difficulty')
    view_distribution(approach_rate_list, 'approach_rate_list')


def mel_from_audio():
    # audio_data, sr = librosa.load(r'./resources/data/bgm/audio.mp3', sr=None)
    audio_data, sr = audio_util.audioread_get_audio_data(
        # r'./resources/data/bgm/audio.mp3',
        r'resources/data/bgm/audio_limelight.mp3',
        ret_tensor=False)
    print('sr')
    print(sr)
    print(audio_data.shape)
    plot_wave(audio_data)
    audio_data = torch.tensor(audio_data, dtype=torch.float)

    n_fft = 1024
    win_length = None
    hop_length = n_fft // 2
    n_mels = 128

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )
    melspec = mel_spectrogram(audio_data)
    melspec = melspec.numpy()
    plot_spectrogram(melspec, 'sr %.f' % sr)
    print(melspec.shape)

    print(np.min(melspec))
    print(np.max(melspec))
    print(np.mean(melspec))
    plt.figure()
    plt.hist(melspec.reshape([-1]))
    plt.show()
    melspec = np.where(melspec == 0, np.finfo(float).eps, melspec)
    melspec = np.log10(melspec)
    print(np.min(melspec))
    print(np.max(melspec))
    print(np.mean(melspec))
    plt.figure()
    plt.hist(melspec.reshape([-1]))
    plt.show()

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr,
        melkwargs={
            'n_fft': n_fft,
            'win_length': win_length,
            'hop_length': hop_length,
            'center': True,
            'pad_mode': "reflect",
            'power': 2.0,
            'norm': "slaney",
            'onesided': True,
            'n_mels': n_mels,
            'mel_scale': "htk",
        }
    )
    mfcc_data = mfcc(audio_data)
    mfcc_data = mfcc_data.numpy()
    print(mfcc_data.shape)
    plot_spectrogram(mfcc_data, 'mfcc')
    print(np.min(mfcc_data))
    print(np.max(mfcc_data))
    print(np.mean(mfcc_data))
    plt.figure()
    plt.hist(mfcc_data.reshape([-1]))
    plt.show()
    # print(audioread.available_backends(True))
    # audio_data, sr = torchaudio.backend.list_audio_backends()
    # print(torchaudio.backend.soundfile_backend.list_audio_backends())
    # =(r'./resources/data/bgm/audio.mp3', ret_tensor=False)

def process_label(label):
    """
    -> circle_heat_value, slider_heat_value, x, y
    """
    L, _ = label.shape
    # bounds in osu! beatmap editor
    x = (label[:, 1] + 180) / (691 + 180)
    y = (label[:, 2] + 82) / (407 + 82)
    # if snap is occupied by a hit_object,
    # noise's value should be almost always within -0.25~+0.25
    heat_value = np.random.randn(2 * L).reshape([L, 2]) / 8
    pos_circle = np.where(label[:, 0] == 1)[0]
    pos_slider = np.where(label[:, 0] == 2)[0]
    heat_value[pos_circle, np.zeros(len(pos_circle), dtype=int)] += 1
    heat_value[pos_slider, np.ones(len(pos_slider), dtype=int)] += 1
    return np.concatenate([heat_value, x[:, np.newaxis], y[:, np.newaxis]], axis=1)


def view_ho_meta_subseq():
    root_dir = r'/home/data1/xiezheng/osu_mapper/preprocessed/meta'
    info_root_dir = r'/home/data1/xiezheng/osu_mapper/preprocessed/info'
    subseq_len = 2560
    num_subseq = 0

    idx_low_slider_occupy = set()
    slider_occupy_thresh = 0.001
    idx_low_circle_count = set()
    circle_count_thresh = 0.0001
    idx_bad = set()

    all_sample_len = []
    all_circle_count = []
    all_slider_occupy = []

    all_beat_divisor = []
    all_diff_circle_count = []
    all_diff_slider_occupy = []
    all_diff_circle_count_p = []
    all_diff_slider_occupy_p = []

    rand_inst = np.random.RandomState(404)
    all_files = os.listdir(root_dir)

    for sample_ho_meta_filename in tqdm(all_files[:5000]):
        filepath = os.path.join(root_dir, sample_ho_meta_filename)
        with open(filepath, 'rb') as f:
            sample_ho_meta, beat_divisor = pickle.load(f)
            all_beat_divisor.append(beat_divisor)
        info_filepath = os.path.join(info_root_dir, sample_ho_meta_filename)
        with open(info_filepath, 'rb') as f:
            sample_len, beatmapsetid, start_frame, end_frame, tps = pickle.load(f)
        sample_len = sample_ho_meta.shape[1]
        all_sample_len.append(sample_len)
        in_sample_subseq_num = sample_len // subseq_len
        rand_end = sample_len - subseq_len
        start_list = rand_inst.randint(start_frame, rand_end, size=[in_sample_subseq_num])

        sample_circle_count = np.mean(sample_ho_meta[0])
        sample_slider_occupy = np.mean(sample_ho_meta[3])

        for i, start_pos in enumerate(start_list):
            subseq_idx = i + num_subseq
            subseq_ho_meta = sample_ho_meta[:, start_pos:start_pos+subseq_len]

            # circle count, slider count, spinner count, slider occupied, spinner occupied
            subseq_circle_count = np.mean(subseq_ho_meta[0])
            subseq_slider_occupy = np.mean(subseq_ho_meta[3])

            all_circle_count.append(subseq_circle_count)
            all_slider_occupy.append(subseq_slider_occupy)

            all_diff_circle_count.append(abs(sample_circle_count - subseq_circle_count))
            all_diff_slider_occupy.append(abs(sample_slider_occupy - subseq_slider_occupy))

            if sample_circle_count > 0:
                all_diff_circle_count_p.append(abs(sample_circle_count - subseq_circle_count) / sample_circle_count)
            if sample_slider_occupy > 0:
                all_diff_slider_occupy_p.append(abs(sample_slider_occupy - subseq_slider_occupy) / sample_slider_occupy)

            if subseq_circle_count < circle_count_thresh:
                idx_low_circle_count.add(subseq_idx)

            if subseq_slider_occupy < slider_occupy_thresh:
                idx_low_slider_occupy.add(subseq_idx)

            if subseq_slider_occupy < slider_occupy_thresh and subseq_circle_count < circle_count_thresh:
                idx_bad.add(subseq_idx)

        num_subseq += len(start_list)

    save_hist(all_sample_len, 'all_sample_len')
    save_hist(all_slider_occupy, 'all_slider_occupy', (0, 0.7))
    save_hist(all_circle_count, 'all_circle_count', (0, 0.1))

    save_hist(all_diff_circle_count, 'all_diff_circle_count', (0, 0.1))
    save_hist(all_diff_slider_occupy, 'all_diff_slider_occupy', (0, 0.7))
    save_hist(all_diff_circle_count_p, 'all_diff_circle_count_p', (0, 1))
    save_hist(all_diff_slider_occupy_p, 'all_diff_slider_occupy_p', (0, 1))

    print(Counter(all_beat_divisor))

    # 3d
    fig = plt.figure()
    plt.hist2d(all_circle_count, all_slider_occupy, bins=200, range=((0, 0.1), (0, 0.7)))
    plt.title('hist2d')
    plt.savefig(
        os.path.join(
            r'/home/data1/xiezheng/osu_mapper/vis',
            'hist2d.jpg'
        )
    )

    print(num_subseq)
    print(len(idx_low_circle_count))
    print(len(idx_low_slider_occupy))
    print(len(idx_bad))


def save_hist(values, title, rg=None):
    fig = plt.figure()
    plt.hist(values, bins=200, range=rg)
    plt.title(title)
    plt.savefig(
        os.path.join(
            r'/home/data1/xiezheng/osu_mapper/vis',
            title + '.jpg'
        )
    )


def np_statistics(arr):
    print(np.mean(arr), np.min(arr), np.max(arr))


def check_signal():

    with open(
        r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\processed_v5\label\999944.pkl',
        'rb'
    ) as f:
        label = pickle.load(f)
        snap_type, x_pos_seq, y_pos_seq = label.T
        print(snap_type)
        print(len(snap_type))
    # with open(
    #     r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\processed\info\40285.pkl',
    #     'rb'
    # ) as f:
    #     sample_len, beatmapsetid, prelude_end_pos = pickle.load(f)
    # print(prelude_end_pos)
    # label = label[prelude_end_pos:prelude_end_pos + 2560]
    # for signal, name in zip(
    #         [
    #             x_pos_seq,
    #             y_pos_seq
    #         ],
    #         # label.T,
    #         [
    #             'cursor_x', 'cursor_y'
    #             # 'circle_hit', 'slider_hit', 'spinner_hit', 'cursor_x', 'cursor_y'
    #         ]
    # ):
    #     print(np.min(signal), np.max(signal), np.mean(signal))
    #     fig_array, fig = plt_util.plot_signal(signal, name,
    #                                  save_path=None,
    #                                  show=True)
    # influence of noise
    from nn.dataset import feeder_embedding
    from postprocess import embedding_decode

    counter_path = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\vis\counter.pkl'
    embedding_path = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\pretrained_models\acgan_embeddingv1_20231006_g0.000003_d0.000003_test\embedding_center-1_-1.pkl'
    decoder = embedding_decode.EmbeddingDecode(
        embedding_path, 8, counter_path
    )

    feeder_args = {'save_dir': r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\processed_v5',
                         'embedding_path': r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\pretrained_models\acgan_embeddingv1_20231006_g0.000003_d0.000003_test\embedding_center-1_-1.pkl',
                         'subseq_snaps': 32 * 8,
                         'random_seed': 404,
                         'use_random_iter': True,
                         'take_first': 1,
                         'pad': False,
                         'beat_divisor': 8,
                         }
    feeder = feeder_embedding.SubseqFeeder(**feeder_args)
    feeder.set_noise_level(0)
    data, [label, embedding], meta = feeder[0]
    print('noise 0')
    vis_subseq(label, embedding, decoder)
    feeder.set_noise_level(0.5)
    data, [label, embedding], meta = feeder[0]
    print('noise 0.5')
    vis_subseq(label, embedding, decoder)


def vis_subseq(label, embedding, decoder):
    for signal, name in zip(
            [
                label[:, 0],
                label[:, 1],
            ],
            # label.T,
            [
                'cursor_x', 'cursor_y'
                # 'circle_hit', 'slider_hit', 'spinner_hit', 'cursor_x', 'cursor_y'
            ]
    ):
        # print(np.min(signal), np.max(signal), np.mean(signal))
        fig_array, fig = plt_util.plot_signal(signal, name,
                                     save_path=None,
                                     show=True)
    print(decoder.decode(embedding))


def count_beat_label_seq():
    beat_divisor = 8
    label_idx_to_beat_label_seq = list(itertools.product(*[(0, 1, 2, 3) for _ in range(beat_divisor)]))
    print(len(label_idx_to_beat_label_seq))
    beat_label_seq_to_label_idx = {seq: idx for idx, seq in enumerate(label_idx_to_beat_label_seq)}
    all_label_idx = []
    # label_idx_dir = r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/label_idx'
    # for label_idx_filename in tqdm(os.listdir(label_idx_dir)):
    #     label_idx_filepath = os.path.join(label_idx_dir, label_idx_filename)
    #     with open(label_idx_filepath, 'rb') as f:
    #         label_idx = pickle.load(f)
    #         all_label_idx.append(label_idx)
    # with open(r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/all_label_idx.pkl', 'wb') as f:
    #     pickle.dump(all_label_idx, f)
    # cnt = Counter(itertools.chain.from_iterable(all_label_idx))
    with open(r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\vis\counter.pkl', 'rb') as f:
        cnt = pickle.load(f)
    print(len(cnt))
    # with open(r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl', 'wb') as f:
    #     pickle.dump(cnt, f)
    most_common = cnt.most_common(10)
    print(most_common)
    for k, v in most_common:
        print(label_idx_to_beat_label_seq[k], v)
    label_idxs, all_occurrences = list(zip(*cnt.most_common(None)))
    fig = plt.figure()
    for k, v in zip(label_idxs[-10:], all_occurrences[-10:]):
        print(label_idx_to_beat_label_seq[k], v)
    print(all_occurrences)
    plt.bar(np.arange(len(all_occurrences)), all_occurrences)
    plt.savefig(r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\vis\occurences.jpg')
    plt.show()


def view_word_embedding():
    # embedding_filepath = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\result\word2vec_skipgramv1_0.1_constlr_dim_16\embedding_center.pkl'
    # embedding_filepath = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\result\word2vec_skipgramv1_0.1_constlr_dim_16_early_stop\embedding_context8_-1.pkl'
    # embedding_filepath = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\result\word2vec_skipgramv1_0.1_constlr_dim_16_early_stop\embedding_context-1_-1.pkl'
    # embedding_filepath = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\result\word2vec_skipgramv1_0.1_constlr_dim_16_early_stop\embedding_context-1_-1.pkl'
    # counter_path = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\vis\counter.pkl'
    embedding_filepath = r'/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl'
    counter_path = r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl'
    with open(embedding_filepath, 'rb') as f:
        embedding = pickle.load(f)
    with open(counter_path, 'rb') as f:
        cnt = pickle.load(f)
    existent_label_idxs = np.array(list(cnt.keys()))
    existent_embedding = embedding[existent_label_idxs]
    pairwise_cosine_dist = [
        np.sum(existent_embedding[i] * existent_embedding[j]) / np.linalg.norm(existent_embedding[i]) / np.linalg.norm(existent_embedding[j])
        for i in range(len(existent_embedding)) for j in range(i+1, len(existent_embedding))
    ]
    print(pairwise_cosine_dist)
    min_cosine_dist = np.min(pairwise_cosine_dist)
    max_cosine_dist = np.max(pairwise_cosine_dist)
    mean_cosine_dist = np.mean(pairwise_cosine_dist)
    print(min_cosine_dist, max_cosine_dist, mean_cosine_dist)

    embedding_norm = np.linalg.norm(existent_embedding, axis=1)
    min_embedding_norm = np.min(embedding_norm)
    max_embedding_norm = np.max(embedding_norm)
    mean_embedding_norm = np.mean(embedding_norm)
    print(min_embedding_norm, max_embedding_norm, mean_embedding_norm)

    normed_embedding = embedding / mean_embedding_norm
    normed_embedding_filepath = r'/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl'
    with open(normed_embedding_filepath, 'wb') as f:
        pickle.dump(normed_embedding, f)
        print('saved to %s' % normed_embedding_filepath)


def count_snap_labels_in_hit_signal():
    root_dir = r'/home/data1/xiezheng/osu_mapper/result/acgan_embeddingv3_20231014_g0.0001_d0.0001_grad_clip_norm1/output'
    all_epoch_count = [0 for _ in range(len(os.listdir(root_dir)))]
    for epoch_folder in tqdm(os.listdir(root_dir)):
        count_tgt = os.path.join(
            root_dir, epoch_folder, 'run1', 'hit_signal.txt'
        )

        with open(count_tgt, 'r') as f:
            hit_signal_str = f.readline()
            counter = Counter(hit_signal_str)
            all_epoch_count[int(epoch_folder)] = counter['3']

    print(all_epoch_count)
    plt.plot(np.arange(len(all_epoch_count)), all_epoch_count)
    plt.savefig(
        os.path.join(
            r'/home/data1/xiezheng/osu_mapper/vis',
            'snap_labels_count_3' + '.jpg'
        )
    )


def star_hist():
    meta_dir = r'/home/data1/xiezheng/osu_mapper/preprocessed_v5/meta'
    all_star = []
    all_cs = []
    for filename in tqdm(os.listdir(meta_dir)):
        filepath = os.path.join(meta_dir, filename)
        with open(filepath, 'rb') as f:
            star, cs = pickle.load(f)
            all_star.append(star)
            all_cs.append(cs)
    plt.hist(all_star, bins=np.linspace(0, 14, 140).tolist())
    plt.title('star')
    plt.savefig(r'/home/data1/xiezheng/osu_mapper/vis/star_hist.png')
    plt.clf()
    # 3.10023066607676 0.19123955711997528 13.700304400500588
    print(np_statistics(np.array(all_star)))
    plt.hist(all_cs)
    plt.title('cs')
    plt.savefig(r'/home/data1/xiezheng/osu_mapper/vis/cs_hist.png')
    plt.clf()
    # 3.671196325409603 0.0 10.0
    print(np_statistics(np.array(all_cs)))
    with open(r'/home/data1/xiezheng/osu_mapper/vis/stars.pkl', 'wb') as f:
        pickle.dump(all_star, f)
    with open(r'/home/data1/xiezheng/osu_mapper/vis/cs.pkl', 'wb') as f:
        pickle.dump(all_cs, f)


def bpm_hist():
    meta_dir = r'/home/data1/xiezheng/osu_mapper/beatmapsets'
    all_bpm = []
    n_processed = 0
    for filename in tqdm(os.listdir(meta_dir), ncols=80):
        osz_path = os.path.join(meta_dir, filename)
        n_processed += 1

        try:
            osz_file = zipfile.ZipFile(osz_path, 'r')
            all_beatmaps = list(slider.Beatmap.from_osz_file(osz_file).values())
            for bm in all_beatmaps:
                if bm.bpm_min() == bm.bpm_max():
                    all_bpm.append(bm.bpm_min())
        except Exception:
            print('beatmap parse failed %s' % osz_file)
            continue
        if n_processed in [1000, 5000, 10000, 20000]:
            plt.figure()
            plt.hist(all_bpm, bins=50)
            plt.title('bpm')
            plt.savefig(r'/home/data1/xiezheng/osu_mapper/vis/bpm_hist_%d.png' % n_processed)
            plt.clf()
            # 3.671196325409603 0.0 10.0
            print(np_statistics(np.array(all_bpm)))
            with open(r'/home/data1/xiezheng/osu_mapper/vis/bpm_%d.pkl' % n_processed, 'wb') as f:
                pickle.dump(all_bpm, f)

    plt.figure()
    plt.hist(all_bpm, bins=50)
    plt.title('bpm')
    plt.savefig(r'/home/data1/xiezheng/osu_mapper/vis/bpm_hist.png')
    plt.clf()
    # 3.671196325409603 0.0 10.0
    print(np_statistics(np.array(all_bpm)))
    with open(r'/home/data1/xiezheng/osu_mapper/vis/bpm.pkl', 'wb') as f:
        pickle.dump(all_bpm, f)
        
        
def beat_divisor_hist():
    meta_dir = r'/home/data1/xiezheng/osu_mapper/beatmapsets'
    all_beat_divisor = []
    n_processed = 0
    for filename in tqdm(os.listdir(meta_dir), ncols=80):
        osz_path = os.path.join(meta_dir, filename)
        n_processed += 1

        try:
            osz_file = zipfile.ZipFile(osz_path, 'r')
            all_beatmaps = list(slider.Beatmap.from_osz_file(osz_file).values())
            for bm in all_beatmaps:
                all_beat_divisor.append(bm.beat_divisor)
        except Exception:
            print('beatmap parse failed %s' % osz_file)
            continue
        if n_processed in [1000, 5000, 10000, 20000]:

            counter = Counter(all_beat_divisor)
            print(counter)
            num = [0 for _ in range(33)]
            for i in range(33):
                if i in counter:
                    num[i] = counter[i]

            plt.figure()
            plt.bar(np.arange(33), num)
            plt.title('beat_divisor')
            plt.savefig(r'/home/data1/xiezheng/osu_mapper/vis/beat_divisor_hist_%d.png' % n_processed)
            plt.clf()
            # 3.671196325409603 0.0 10.0
            print(np_statistics(np.array(all_beat_divisor)))
            with open(r'/home/data1/xiezheng/osu_mapper/vis/beat_divisor_%d.pkl' % n_processed, 'wb') as f:
                pickle.dump(all_beat_divisor, f)

    counter = Counter(all_beat_divisor)
    print(counter)
    num = [0 for _ in range(33)]
    for i in range(33):
        if i in counter:
            num[i] = counter[i]

    plt.figure()
    plt.bar(np.arange(33), num)
    plt.title('beat_divisor')
    plt.savefig(r'/home/data1/xiezheng/osu_mapper/vis/beat_divisor_hist.png')
    plt.clf()
    # 3.671196325409603 0.0 10.0
    print(np_statistics(np.array(all_beat_divisor)))
    with open(r'/home/data1/xiezheng/osu_mapper/vis/beat_divisor.pkl', 'wb') as f:
        pickle.dump(all_beat_divisor, f)


def seq_len_hist():
    info_dir = r'/home/xiezheng/data/preprocessed_v7/info'
    all_beatmap_nmel_count = []
    n_processed = 0
    for filename in tqdm(os.listdir(info_dir), ncols=80):
        info_path = os.path.join(info_dir, filename)
        n_processed += 1

        with open(info_path, 'rb') as f:
            n_mel = pickle.load(f)[0]
        all_beatmap_nmel_count.append(n_mel)
    # 24963.777327398428 2048 114240
    # -> 390 32 1785
    all_beatmap_nmel_count = np.array(all_beatmap_nmel_count)
    print(np_statistics(all_beatmap_nmel_count))
    with open(r'/home/xiezheng/osu_mapper/resources/vis/all_beatmap_nmel_count.pkl', 'wb') as f:
        pickle.dump(all_beatmap_nmel_count, f)
    plt.figure()
    plt.hist(all_beatmap_nmel_count)
    plt.title('all_beatmap_nmel_count')
    plt.savefig(r'/home/xiezheng/osu_mapper/resources/vis/all_beatmap_nmel.png')
    plt.clf()
    all_beatmap_beat_count = all_beatmap_nmel_count / 8 / 8


def view_label_group_vocab():
    with open() as f:
        pass


if __name__ == '__main__':
    """
    view beatmap total snap number hist
    """
    seq_len_hist()
    """
    view beat_divisor_hist
    """
    # beat_divisor_hist()
    """
    view bpm hist
    """
    # bpm_hist()
    """
    view star hist
    """
    # # view_word_embedding()
    # # star_hist()
    # with open(r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\vis\stars.pkl', 'rb') as f:
    #     all_star = pickle.load(f)
    #
    # def smaller_than(star_ref):
    #     count = 0
    #     for star in all_star:
    #         if star < star_ref:
    #             count += 1
    #     return count / len(all_star)
    #
    # for s in [1, 2, 3, 4, 5, 6, 7, 8]:
    #     print(smaller_than(s))

    """
    """
    # count_snap_labels_in_hit_signal()
    """
    view embedding
    """
    # view_word_embedding()
    """
    count label idx
    """
    # count_beat_label_seq()
    """
    view mel statistics
    """
    # view_ho_meta_subseq()
    # with open(r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\processed\mel\999739.pkl', 'rb') as f:
    #     mel = pickle.load(f)
    # print(mel.shape)
    # print(np_statistics(mel[:40]))
    # # print(mel[:, 40])
    # with open(r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\processed\melv3\999739.pkl', 'rb') as f:
    #     mel = pickle.load(f)
    # print(mel.shape)
    # print(np_statistics(mel[:40]))
    # with open(r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\processed\mel\10314.pkl', 'rb') as f:
    #     mel = pickle.load(f)
    # print(mel.shape)
    # print(np_statistics(mel[:40]))
    # with open(r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\processed\mel\10314.pkl', 'rb') as f:
    #     mel = pickle.load(f)
    # print(mel.shape)
    # print(np_statistics(mel[:40]))
    # print(mel[:, 40])
    # view_ho_meta_subseq()

    # processed = process_label(label_list[0])
    # print(processed)
    # mel_from_audio()
    # assert isinstance(beatmap, slider.Beatmap)
    # try:
    #     all_speed_stars.append(beatmap.speed_stars())
    # except Exception:
    #     continue
    # plt.hist(all_speed_stars)
    # plt.show()
    # print(beatmap.speed_stars)
    # snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
    # label = dataset_util.hitobjects_to_label_with_pos(beatmap, snap_ms=snap_ms)
    # if label is not None:
    #     x, y = label[:, 1], label[:, 2]
    #     pos = np.where(x < 0)
    #     print(pos)
    #     # print(label[pos])
    #     print(label)
    #     input()
