import argparse
import math
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import slider
from BeatNet.BeatNet import BeatNet
from slider import curve

from util import audio_util, beatmap_util
import inference


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', type=str, help=r'Path to audio file to convert.'
    )
    return parser


def _timedelta_to_milliseconds(td):
    return td / timedelta(microseconds=1) / 1000


def _pack_timedelta(td):
    return str(td // timedelta(microseconds=1) // 1000)


def _pack_bool(bool_in):
    return '1' if bool_in else '0'


def _pack_str(str_in):
    return str_in


def _pack_str_list(list_str):
    return ' '.join(list_str)


def _pack_hit_object(hit_object: slider.beatmap.HitObject):
    print(hit_object.double_time)
    if isinstance(hit_object, slider.beatmap.Slider):
        object_params_str = _pack_slider_object_params(hit_object)
    elif isinstance(hit_object, slider.beatmap.Spinner):
        object_params_str = _pack_spinner_object_params(hit_object)
    else:
        # Circle
        return ','.join([str(int(hit_object.position.x)),
                         str(int(hit_object.position.y)),
                         _pack_timedelta(hit_object.time),
                         str(slider.beatmap.Circle.type_code),
                         str(hit_object.hitsound),
                         hit_object.addition])
    hit_object_str = ','.join([str(int(hit_object.position.x)),
                               str(int(hit_object.position.y)),
                               _pack_timedelta(hit_object.time),
                               str(hit_object.type_code),
                               str(hit_object.hitsound),
                               object_params_str,
                               hit_object.addition])
    return hit_object_str

# def _circle_to_str(hit_object: slider.beatmap.Circle):
#     return

def _pack_slider_object_params(hit_object: slider.beatmap.Slider):
    # curve_type = str(type(hit_object.curve))
    # print(curve_type)
    # if 'Catmull' in curve_type:
    #     curve_type_str = 'C'
    # elif 'Linear' in curve_type:
    #     curve_type_str = 'L'
    # elif 'Perfect' in curve_type:
    #     curve_type_str = 'P'
    # else:
    #     curve_type_str = 'B'
    curve_type = str(type(hit_object.curve))
    print(curve_type)
    if isinstance(hit_object, curve.Catmull):
        curve_type_str = 'C'
    elif 'Linear' in curve_type:
        curve_type_str = 'L'
    elif 'Perfect' in curve_type:
        curve_type_str = 'P'
    else:
        curve_type_str = 'B'
    curve_str = [curve_type_str]
    write_points = hit_object.curve.points[1:]
    for point in write_points:
        curve_str.append(str(int(point.x)) + ':' + str(int(point.y)))
    curve_str = '|'.join(curve_str)
    return ','.join([curve_str, str(hit_object.repeat), str(hit_object.length),
                     '|'.join(str(edge_sound) for edge_sound in hit_object.edge_sounds),
                     '|'.join(str(edge_addition) for edge_addition in hit_object.edge_additions)])


def _pack_spinner_object_params(hit_object: slider.beatmap.Spinner):
    return _pack_timedelta(hit_object.end_time)


def _pack_timing_point(timing_point: slider.beatmap.TimingPoint):
    return ','.join([_pack_timedelta(timing_point.offset),
                     str(timing_point.ms_per_beat),
                     str(timing_point.meter),
                     str(timing_point.sample_type),
                     str(timing_point.sample_set),
                     str(timing_point.volume),
                     '1' if timing_point.parent is None else '0',
                     _pack_bool(timing_point.kiai_mode)])


def dump_beatmap_to_path(beatmap: slider.Beatmap, path):
    with open(path, 'w', encoding='utf-8') as f:
        dump_beatmap(beatmap, f)


def dump_beatmap(beatmap: slider.Beatmap, f):
    field_names = [
        # General section fields
        'AudioFilename', 'AudioLeadIn',
        'PreviewTime', 'Countdown',
        'SampleSet', 'StackLeniency', 'Mode',
        'LetterboxInBreaks', 'WidescreenStoryboard',

        # Editor section fields
        'DistanceSpacing', 'BeatDivisor',
        'GridSize', 'TimelineZoom',

        # Metadata section fields
        'Title', 'TitleUnicode',
        'Artist', 'ArtistUnicode',
        'Creator', 'Version',
        'Source', 'Tags',
        'BeatmapID', 'BeatmapSetID',

        # Difficulty section fields
        'HPDrainRate', 'CircleSize',
        'OverallDifficulty', 'ApproachRate',
        'SliderMultiplier', 'SliderTickRate',
    ]
    fields = [
        # General section fields
        beatmap.audio_filename, beatmap.audio_lead_in,
        beatmap.preview_time, beatmap.countdown,
        beatmap.sample_set, beatmap.stack_leniency, beatmap.mode,
        beatmap.letterbox_in_breaks, beatmap.widescreen_storyboard,

        # Editor section fields
        beatmap.distance_spacing, beatmap.beat_divisor,
        beatmap.grid_size, beatmap.timeline_zoom,

        # Metadata section fields
        beatmap.title, beatmap.title_unicode,
        beatmap.artist, beatmap.artist_unicode,
        beatmap.creator, beatmap.version,
        beatmap.source, beatmap.tags,
        beatmap.beatmap_id, beatmap.beatmap_set_id,

        # Difficulty section fields
        beatmap.hp_drain_rate, beatmap.circle_size,
        beatmap.overall_difficulty, beatmap.approach_rate,
        beatmap.slider_multiplier, beatmap.slider_tick_rate,
    ]

    groups = {
        'General': {
            'AudioFilename': (beatmap.audio_filename, str),
            'AudioLeadIn': (beatmap.audio_lead_in, _pack_timedelta),
            'PreviewTime': (beatmap.preview_time, _pack_timedelta),
            'Countdown': (beatmap.countdown, _pack_bool),
            'SampleSet': (beatmap.sample_set, str),
            'StackLeniency': (beatmap.stack_leniency, str),
            'Mode': (beatmap.mode, lambda x: str(int(x))),
            'LetterboxInBreaks': (beatmap.letterbox_in_breaks, _pack_bool),
            'WidescreenStoryboard': (beatmap.widescreen_storyboard, _pack_bool),
        },
        'Editor': {
            'DistanceSpacing': (beatmap.distance_spacing, str),
            'BeatDivisor': (beatmap.beat_divisor, str),
            'GridSize': (beatmap.grid_size, str),
            'TimelineZoom': (beatmap.timeline_zoom, str),
        },
        'Metadata': {
            'Title': (beatmap.title, str),
            'TitleUnicode': (beatmap.title_unicode, str),
            'Artist': (beatmap.artist, str),
            'ArtistUnicode': (beatmap.artist_unicode, str),
            'Creator': (beatmap.creator, str),
            'Version': (beatmap.version, str),
            'Source': (beatmap.source, str),
            'Tags': (beatmap.tags, _pack_str_list),
            'BeatmapID': (beatmap.beatmap_id, str),
            'BeatmapSetID': (beatmap.beatmap_id, str),
        },
        'Difficulty': {
            'HPDrainRate': (beatmap.hp_drain_rate, str),
            'CircleSize': (beatmap.circle_size, str),
            'OverallDifficulty': (beatmap.overall_difficulty, str),
            'ApproachRate': (beatmap.approach_rate, str),
            'SliderMultiplier': (beatmap.slider_multiplier, str),
            'SliderTickRate': (beatmap.slider_tick_rate, str),
        },
        'Events':
            ['// Background and Video events',
             '// Break Periods',
             '// Storyboard Layer 0(Background)',
             '// Storyboard Layer 1(Fail)',
             '// Storyboard Layer 2(Pass)',
             '// Storyboard Layer 3(Foreground)',
             '// Storyboard Layer 4(Overlay)',
             '// Storyboard Sound Samples', ],
        'TimingPoints': beatmap.timing_points,
        'HitObjects': beatmap.hit_objects(),
        'Colours': None,
    }
    f.write('osu file format v14\n\n')

    def dump_group(group, group_items):
        f.write('[' + group + ']\n')
        for field, field_content in group_items.items():
            f.write(field)
            f.write(':')
            f.write(field_content[1](field_content[0]))
            f.write('\n')
        f.write('\n')

    def dump_events(group, group_items):
        f.write('[' + group + ']\n')
        for item in group_items:
            f.write(item)
            f.write('\n')
        f.write('\n')

    def dump_colours(group, group_items):
        f.write('[' + group + ']\n')
        # for item in group_items:
        #     f.write(item)
        #     f.write('\n')
        f.write('\n')

    def dump_timing_points(group, group_items):
        f.write('[' + group + ']\n')
        for item in group_items:
            f.write(_pack_timing_point(item))
            f.write('\n')
        f.write('\n')

    def dump_hit_objects(group, group_items):
        f.write('[' + group + ']\n')
        for item in group_items:
            f.write(_pack_hit_object(item))
            f.write('\n')
        f.write('\n')

    for group, group_dict in groups.items():
        if group == 'Events':
            dump_events(group, group_dict)
        elif group == 'HitObjects':
            dump_hit_objects(group, group_dict)
        elif group == 'TimingPoints':
            dump_timing_points(group, group_dict)
        elif group == 'Colours':
            pass
            # dump_colours(group, group_dict)
        else:
            dump_group(group, group_dict)


def num_into_bin(num_list, bin_num=None, bin_size=None, start_bin_min=None, end_bin_max=None):
    """
    Assign numbers to bins

    :param num_list:
    :param bin_num: neglected if bin_size is given
    :param bin_size:
    :param start_bin_min:
    :param end_bin_max:
    :return:
    """
    num_list_min = np.min(num_list)
    num_list_max = np.max(num_list)
    count_max_item_in_last_bin = True
    if start_bin_min is None:
        start_bin_min = num_list_min
    else:
        if num_list_min < start_bin_min:
            print('num smaller than start_bin_min is not counted.')
    if end_bin_max is None:
        end_bin_max = num_list_max
    else:
        if num_list_max > end_bin_max:
            print('num larger than end_bin_max is not counted.')
            count_max_item_in_last_bin = False
    if bin_size is None:
        if bin_num is None:
            print('Should assign at least one in between bin_size and bin_num!')
            print('bin_num falling back to 1!')
            bin_num = 1
            bin_size = end_bin_max - start_bin_min
        else:
            bin_size = (end_bin_max - start_bin_min) / bin_num
    else:
        if bin_num is not None:
            print('Taking bin_size as reference.')
        bin_num = math.ceil((end_bin_max - start_bin_min) / bin_size)
        end_bin_max = start_bin_min + bin_num * bin_size
    bin_split = [start_bin_min + i * bin_size for i in range(bin_num)]
    bin_split.append(end_bin_max)
    bin_occurrence = [0 for _ in range(bin_num)]
    bin_item_idx = [[] for _ in range(bin_num)]
    for num_idx, num in enumerate(num_list):
        bin_idx_raw = (num - start_bin_min) / bin_size
        if bin_idx_raw > bin_num or bin_idx_raw < 0:
            continue
        if count_max_item_in_last_bin and bin_idx_raw == bin_num:
            bin_idx_raw = bin_num - 1
        bin_idx = math.floor(bin_idx_raw)
        bin_occurrence[bin_idx] += 1
        bin_item_idx[bin_idx].append(num_idx)
    return bin_occurrence, bin_split, bin_item_idx


def plot_label_distribution(label_list, plot_binary=False):
    max_label = np.max(label_list)
    min_label = np.min(label_list)
    label_type_num = max_label - min_label + 1
    print('max_label:\t\t' + str(max_label))
    print('min_label:\t\t' + str(min_label))
    print('label_type_num:\t' + str(label_type_num))
    label_occurrence = [0 for _ in range(label_type_num)]
    for i in range(len(label_list)):
        label_occurrence[label_list[i]] += 1
    print('label_occurrence:\t' + str(label_occurrence))
    if plot_binary:
        label_type = [0, 1]
        total_1s = 0
        for i in range(1, label_type_num):
            total_1s += label_occurrence[i]
        binary_label_occurrence = [label_occurrence[0], total_1s]
        print('binary_label_occurrence:\t' + str(binary_label_occurrence))
        plt.bar(label_type, binary_label_occurrence)
        plt.show()
        return binary_label_occurrence
    else:
        label_type = list(range(label_type_num))
        plt.bar(label_type, label_occurrence)
        plt.show()
        return label_occurrence


def thresh_bin_mean(itv_list, bin_size=0.1, thresh=0.3):
    bin_occurrence, bin_split, bin_item_idx = num_into_bin(itv_list, bin_size=bin_size)
    # print(bin_occurrence)
    # print(bin_split)
    # print(bin_item_idx)
    max_bin_idx = np.argmax(bin_occurrence)
    over_thresh_bin_idx = np.where(np.array(bin_occurrence) > bin_occurrence[max_bin_idx] * thresh)[0]
    # print(over_thresh_bin_idx)
    items_idx_over_thresh = []
    for idx in over_thresh_bin_idx:
        items_idx_over_thresh.extend(bin_item_idx[idx])
    items_over_thresh = np.array(itv_list)[items_idx_over_thresh]
    plt.bar(bin_split[1:], bin_occurrence)
    plt.show()
    v = np.mean(items_over_thresh)
    return v


def extract_bpm(audio_file_path):
    """
    returns bpm, first beat microsecond, last beat microsecond
    """
    estimator = BeatNet(1, mode='online', inference_model='PF', plot=['activations'], thread=False)
    output = estimator.process(audio_file_path)
    itv = np.diff(output[:, 0])
    bpm = 60 / thresh_bin_mean(itv, 0.02)
    first_beat = output[0, 0]
    last_beat = output[-1, 0]
    return bpm, first_beat * 1000000, last_beat * 1000000


class BeatmapGenerator:
    def __init__(self, config_path, model_path, device='cpu'):
        self.inference = inference.Inference(config_path, model_path)

    def generate_beatmap(self, audio_file_path, speed_stars, out_path, audio_info_path=None):
        if audio_info_path is not None:
            # load audio info (bpm, first_beat, last_beat) from file if possible
            if os.path.exists(audio_info_path):
                with open(audio_info_path, 'r') as f:
                    line = f.readline()
                    s = line.split(',')
                    bpm, first_beat, last_beat = [float(part) for part in s]
            else:
                bpm, first_beat, last_beat = extract_bpm(audio_file_path)
                with open(audio_info_path, 'w') as f:
                    f.write(','.join([str(bpm), str(first_beat), str(last_beat)]))
        else:
            bpm, first_beat, last_beat = extract_bpm(audio_file_path)

        print('bpm %f, first_beat %f, last_beat %f' % (bpm, first_beat, last_beat))
        label = self.inference.run_inference(audio_file_path, speed_stars, bpm,
                                             start_time=first_beat, end_time=last_beat)
        beatmap = self.label_to_beatmap(label, speed_stars, bpm, start_time=first_beat, end_time=last_beat)
        beatmap.write_path(out_path)

    @staticmethod
    def label_to_beatmap(label, speed_stars, bpm, start_time, end_time, snap_divisor=8):
        beatmap = beatmap_util.get_empty_beatmap()
        beatmap.bpm_min = bpm
        beatmap.bpm_max = bpm
        beatmap.speed_stars = speed_stars
        ms_per_beat = 60000 / bpm
        snap_per_microsecond = bpm * snap_divisor / 60000000
        total_snap_num = len(label)
        print('label')
        print(label)
        print('total_snap_num')
        print(total_snap_num)
        print('total snap num')
        print((end_time - start_time) * snap_per_microsecond)
        left, right, top, bottom = 0, 640, 0, 480
        beatmap.timing_points.append(slider.beatmap.TimingPoint(
            timedelta(microseconds=start_time),
            ms_per_beat,
            0, 0, 0, 100, None, False
        ))
        for snap_idx, snap_result in enumerate(label):
            if snap_result == 1:
                beatmap._hit_objects.append(slider.beatmap.Circle(
                    slider.Position(0, 0),
                    timedelta(microseconds=snap_idx / snap_per_microsecond + start_time),
                    0))
        return beatmap


if __name__ == '__main__':
    generator = BeatmapGenerator(
        r'../resources/config/cnnv1.yaml',
        r'../resources/result/cnnv1/0.1/model_epoch5.pt'
    )
    # audio_file_path = r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\bgm\audio.mp3'
    audio_file_path = r'C:\Users\asus\AppData\Local\osu!\Songs\869801 Meramipop - Sacrifice\audio.mp3'
    out_path = r'..\resources\data\bgm\out.osu'
    audio_info_path = r'..\resources\data\bgm\audio_info.txt'
    generator.generate_beatmap(audio_file_path, 3, out_path, audio_info_path)
    # path = r''
    # p = get_parser()
    # p.parse_args([path])
    # beatmap = slider.Beatmap.from_path(
    #     # r'C:\Users\asus\AppData\Local\osu!\Songs\1591723 solfa feat Ceui - primal\solfa feat. Ceui - primal (Shikibe Mayu) [Memories of Spring].osu'
    #     r'C:\Users\asus\coding\python\slider\slider\example_data\beatmaps\AKINO from bless4 & CHiCO with HoneyWorks - MIIRO vs. Ai no Scenario (monstrata) [Tatoe].osu'
    # )
    # print(beatmap.hit_objects()[12])
    # dump_beatmap_to_path(
    #     beatmap,
    #     r'C:\Users\asus\AppData\Local\osu!\Songs\beatmap-637829914631239489-audio\solfa feat. Ceui - primal (Shikibe Mayu) [Memories of Spring].osu'
    # )
    # for ho in beatmap.hit_objects():
    #     if isinstance(ho, slider.beatmap.Slider):
    #         print(ho.curve)
    #         print(type(ho.curve))
    # with open(path, 'r') as f:
    #     dump_beatmap(beatmap, f)

    # a = timedelta(days=23)
    # print(_timedelta_to_milliseconds(a))
    # print(_timedelta_to_milliseconds_str(a))
    # print(partial(timedelta, days=23)(a))
