import pickle
import types

import slider
import os
from sklearn import model_selection
import torchaudio
import shutil

from util import audio_util


class OSUSongsDir:
    """
    Osu! Songs directory
    """
    def __init__(self, songs_dir=r'C:\Users\asus\AppData\Local\osu!\Songs'):
        self.songs_dir = songs_dir

    def beatmaps(self):
        """
        Walk through Osu! Songs directory.
        """
        for beatmapset_dirname in os.listdir(self.songs_dir):
            beatmapset_dir_path = os.path.join(self.songs_dir, beatmapset_dirname)
            osu_filename_list = os.listdir(beatmapset_dir_path)
            # all .osu files under a beatmapset dir
            osu_filename_list = [filename for filename in osu_filename_list if filename.endswith('.osu')]
            for osu_filename in osu_filename_list:
                osu_file_path = os.path.join(beatmapset_dir_path, osu_filename)
                yield beatmapset_dirname, beatmapset_dir_path, osu_filename, osu_file_path

    def gen_index_file(self, index_file_path=r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\local.pkl'):
        """
        Generate an index file containing information about all osu! file path under OSUSongsDir
        and their corresponding audio file path.
        """
        audio_osu_list = []
        for beatmapset_dirname, beatmapset_dir_path, osu_filename, osu_file_path in self.beatmaps():
            try:
                beatmap = slider.Beatmap.from_path(osu_file_path)
            except UnicodeDecodeError:
                print('slider failed to decode %s!' % osu_file_path)
                continue
            # print(beatmap.beatmap_set_id)
            # print(osu_filename)
            audio_file_path = os.path.join(beatmapset_dir_path, beatmap.audio_filename)
            audio_osu_list.append([audio_file_path, osu_file_path])
        if not os.path.exists(os.path.dirname(index_file_path)):
            os.makedirs(os.path.dirname(index_file_path))
        print('%d valid beatmaps in total.' % len(audio_osu_list))
        with open(index_file_path, 'wb') as f:
            pickle.dump(audio_osu_list, f)


class DataFilter:
    def __init__(self, data):
        self.data = data

    def keep_satisfied(self, params, target_path):
        """
        Delete from list if not satisfying conditions specified in params.
        If params is a dict, whose keys corresponding to attr of slider.Beatmap object.
        If params is a function, it should take in a slider.Beatmap object as input and
        return True/False if condition is/is not satisfied.
        """
        satisfied_index = self.data.get_satisfied(params)
        print('%d beatmaps satisfying condition.' % len(satisfied_index))
        data = self.data.from_index(self.data, satisfied_index, target_path)
        data.save()
        return data


class FoldDivider:
    def __init__(self, data):
        self.data = data

    def div_folds(self, fold_dir, folds=5, shuffle=False):
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        kf = model_selection.KFold(folds, shuffle=shuffle)
        fold = 1
        for train_index, test_index in kf.split(self.data.audio_osu_list):
            print('generating folds %d data...' % fold)
            self.data.from_index(self.data, train_index, os.path.join(fold_dir, 'train%d.pkl' % fold)).save()
            self.data.from_index(self.data, test_index, os.path.join(fold_dir, 'test%d.pkl' % fold)).save()
            fold += 1


def download_beatmap_with_specifications(lib, since, game_mode=slider.game_mode.GameMode.standard, api_key=slider.client.Client.DEFAULT_API_URL):
    client = slider.client.Client(lib, api_key)
    beatmapresult_list = client.beatmap(since=since, game_mode=game_mode)
    for beatmapresult in beatmapresult_list:
        print(beatmapresult.beatmap_id)
        beatmap = beatmapresult.beatmap(save=True)


def local_to_dataset(songs_dir=r'C:\Users\asus\AppData\Local\osu!\Songs', folds=5):
    """
    Extract useful information from local osu! game Songs directory to generate a dataset.
    """
    kf = model_selection.KFold(n_splits=folds, shuffle=False)
    for beatmapset_dirname in os.listdir(songs_dir):
        beatmapset_dir_path = os.path.join(songs_dir, beatmapset_dirname)
        osu_filename_list = os.listdir(beatmapset_dir_path)
        # all .osu files under a beatmapset dir
        osu_filename_list = [filename for filename in osu_filename_list if filename.endswith('.osu')]
        osu_file_path_list = [os.path.join(beatmapset_dir_path, filename) for filename in osu_filename_list]
        for osu_file_path in osu_file_path_list:
            beatmap = slider.Beatmap.from_path(osu_file_path)
            audio_file_path = os.path.join(beatmapset_dir_path, beatmap.audio_filename)
            torchaudio.backend.list_audio_backends()


# def fetch_audio_osu(songs_dir=r'C:\Users\asus\AppData\Local\osu!\Songs'):
#     """
#     Fetch audio files and corresponding osu! files from local osu! game Songs directory,
#     audio files are converted to wav.
#     """
#     for beatmapset_dirname in os.listdir(songs_dir):
#         beatmapset_dir_path = os.path.join(songs_dir, beatmapset_dirname)
#         osu_filename_list = os.listdir(beatmapset_dir_path)
#         # all .osu files under a beatmapset dir
#         osu_filename_list = [filename for filename in osu_filename_list if filename.endswith('.osu')]
#         osu_file_path_list = [os.path.join(beatmapset_dir_path, filename) for filename in osu_filename_list]
#         for osu_file_path in osu_file_path_list:
#             beatmap = slider.Beatmap.from_path(osu_file_path)
#             audio_file_path = os.path.join(beatmapset_dir_path, beatmap.audio_filename)
#             target_audio_file_path = os.path.join()
#             audio_util.audio_to(audio_file_path, )


def fetch_audio_osu(songs_dir=r'C:\Users\asus\AppData\Local\osu!\Songs',
                    audio_osu_dir=r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\audio_osu'):
    """
    Fetch audio files and corresponding osu! files from local osu! game Songs directory,
    audio files are converted to wav, all files renamed with beatmap_id.
    """
    audio_dir = os.path.join(audio_osu_dir, 'wav')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    osu_dir = os.path.join(audio_osu_dir, 'osu')
    if not os.path.exists(osu_dir):
        os.makedirs(osu_dir)
    audio_osu_pairs = []
    for beatmapset_dirname, beatmapset_dir_path, osu_filename, osu_file_path in OSUSongsDir(songs_dir).beatmaps():
        beatmap = slider.Beatmap.from_path(osu_file_path)
        target_osu_file_path = os.path.join(osu_dir, str(beatmap.beatmap_id) + '.osu')
        shutil.copy(osu_file_path, target_osu_file_path)
        target_audio_file_path = os.path.join(audio_dir, str(beatmap.beatmap_id) + '.wav')
        if not os.path.exists(target_audio_file_path):
            audio_file_path = os.path.join(beatmapset_dir_path, beatmap.audio_filename)
            audio_util.audio_to(audio_file_path, target_audio_file_path)
        audio_osu_pairs.append((target_audio_file_path, target_osu_file_path))
    with open(os.path.join(audio_osu_dir, 'audio_osu_pairs.pkl'), 'wb') as f:
        pickle.dump(audio_osu_pairs, f)
