import os
import pickle
import random
import types

import slider
from sklearn import model_selection

from util import audio_util


class AudioOsuDataFoldDivider:
    def __init__(self, data):
        self.data = data

    def div_folds(self, fold_dir, folds=5, shuffle=False):
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        kf = model_selection.KFold(folds, shuffle=shuffle)
        fold = 1
        for train_index, test_index in kf.split(self.data.audio_osu_list):
            print('generating folds %d cond_data...' % fold)
            self.data.from_index(self.data, train_index, os.path.join(fold_dir, 'train%d.pkl' % fold)).save()
            self.data.from_index(self.data, test_index, os.path.join(fold_dir, 'test%d.pkl' % fold)).save()
            fold += 1


class AudioOsuData:
    def __init__(self, audio_osu_list=None, beatmaps=None, index_file_path=None, beatmap_obj_file_path=None):
        self.index_file_path = index_file_path
        self.beatmap_obj_file_path = beatmap_obj_file_path
        self.audio_osu_list = audio_osu_list
        self.beatmaps = beatmaps

    @classmethod
    def from_index(cls, inst_ref, index, index_file_path=None, beatmap_obj_file_path=None):
        audio_osu_list = inst_ref.audio_osu_list_by_index(index)
        beatmaps = inst_ref.beatmaps_by_index(index)
        inst = cls(audio_osu_list, beatmaps, index_file_path, beatmap_obj_file_path)
        if beatmap_obj_file_path is None and index_file_path is not None:
            inst.beatmap_obj_file_path = inst.get_beatmap_obj_file_path(index_file_path)
        return inst

    def save(self, index_file_path=None, beatmap_obj_file_path=None, audio_osu_list=None, beatmaps=None):
        """
        If called with certain param as None, instance's corresponding param is used.
        """
        self.save_index_file(index_file_path, audio_osu_list)
        self.save_beatmap_obj(beatmap_obj_file_path, beatmaps)

    def shuffle(self):
        assert self.beatmaps is not None
        assert self.audio_osu_list is not None
        zipped = list(zip(self.audio_osu_list, self.beatmaps))
        random.shuffle(zipped)
        self.beatmaps, self.audio_osu_list = list(zip(*zipped))

    @classmethod
    def from_path(cls, index_file_path=r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\local.pkl',
                 beatmap_obj_file_path=None):
        inst = cls(index_file_path=index_file_path, beatmap_obj_file_path=beatmap_obj_file_path)
        if beatmap_obj_file_path is None:
            inst.beatmap_obj_file_path = inst.get_beatmap_obj_file_path()
        with open(index_file_path, 'rb') as f:
            inst.audio_osu_list = pickle.load(f)
        inst.load_beatmap_obj()
        if not os.path.exists(inst.beatmap_obj_file_path):
            inst.save_beatmap_obj()
        print('beatmaps loaded')
        print('%d beatmaps in total' % len(inst.beatmaps))
        return inst

    def get_beatmap_obj_file_path(self, index_file_path=None):
        if index_file_path is None:
            assert self.index_file_path is not None
            index_file_path = self.index_file_path
        index_filename = os.path.basename(self.index_file_path)
        return os.path.join(os.path.dirname(index_file_path), index_filename[:-4] + '_beatmaps' + index_filename[-4:])

    def get_sample_rate_list_by_index(self, index):
        if isinstance(index, int):
            audio_file_path = self.audio_osu_list[index][0]
            return audio_util.get_audio_attr(audio_file_path, 'sample_rate')
        else:
            return [self.get_sample_rate_list_by_index(i) for i in index]

    def get_satisfied(self, params):
        """
        Get item index in self.audio_osu_list satisfying conditions specified in params.
        If params is a dict, its keys should correspond to attr of slider.Beatmap object.
        If params is a function, it should take in a slider.Beatmap object as input and
        return True/False if condition is/is not satisfied.
        """
        satisfied_index = []
        for i, audio_osu_item in enumerate(self.audio_osu_list):
            # audio_file_path, osu_file_path = audio_osu_item
            # beatmap = slider.Beatmap.from_path(osu_file_path)
            beatmap = self.beatmaps[i]
            if isinstance(params, dict):
                all_satisfied = True
                for k, v in params.items():
                    p = getattr(beatmap, k)
                    if isinstance(v, (list, tuple)):
                        if isinstance(p, float):
                            # if attr value is float, v is interpreted as an interval
                            assert len(v) == 2
                            if v[0] > p or v[1] < p:
                                all_satisfied = False
                                break
                        elif p not in v:
                            # if attr value is int, v is interpreted as a list of valid choices
                            all_satisfied = False
                            break
                    else:
                        # if v is not list or tuple, compare attr value with v directly
                        if p != v:
                            all_satisfied = False
                            break
                if all_satisfied:
                    satisfied_index.append(i)
            elif isinstance(params, types.FunctionType):
                if params(beatmap):
                    satisfied_index.append(i)
            else:
                raise ValueError('params has wrong type')
        return satisfied_index

    def beatmaps_by_index(self, index):
        assert self.beatmaps is not None
        if isinstance(index, int):
            return self.beatmaps[index]
        else:
            return [self.beatmaps[i] for i in index]

    def audio_osu_list_by_index(self, index):
        assert self.audio_osu_list is not None
        if isinstance(index, int):
            return self.audio_osu_list[index]
        else:
            return [self.audio_osu_list[i] for i in index]

    def get_attr_list(self, attr):
        attr_list = []
        for beatmap in self.beatmaps:
            attr_inst = getattr(beatmap, attr)
            if isinstance(attr_inst, types.MethodType):
                attr_inst = attr_inst()
            attr_list.append(attr_inst)
        return attr_list

    def save_beatmap_obj(self, beatmap_obj_file_path=None, beatmaps=None):
        if beatmaps is None:
            beatmaps = self.beatmaps
        if beatmap_obj_file_path is None:
            beatmap_obj_file_path = self.beatmap_obj_file_path
        if not os.path.exists(os.path.dirname(beatmap_obj_file_path)):
            os.makedirs(os.path.dirname(beatmap_obj_file_path))
        with open(beatmap_obj_file_path, 'wb') as f:
            pickle.dump(beatmaps, f)

    def save_index_file(self, index_file_path=None, audio_osu_list=None):
        if audio_osu_list is None:
            audio_osu_list = self.audio_osu_list
        if index_file_path is None:
            index_file_path = self.index_file_path
        if not os.path.exists(os.path.dirname(index_file_path)):
            os.makedirs(os.path.dirname(index_file_path))
        with open(index_file_path, 'wb') as f:
            pickle.dump(audio_osu_list, f)

    def load_beatmap_obj(self, beatmap_obj_file_path=None):
        if self.beatmap_obj_file_path is None:
            self.beatmap_obj_file_path = self.get_beatmap_obj_file_path(self.index_file_path)
        if beatmap_obj_file_path is None:
            beatmap_obj_file_path = self.beatmap_obj_file_path
        if not os.path.exists(beatmap_obj_file_path):
            print('loading beatmap objects from .osu files...')
            self.beatmaps = [slider.Beatmap.from_path(audio_osu_item[1]) for audio_osu_item in self.audio_osu_list]
        else:
            with open(beatmap_obj_file_path, 'rb') as f:
                self.beatmaps = pickle.load(f)

    def __len__(self):
        return len(self.audio_osu_list)
