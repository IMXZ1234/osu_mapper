import itertools
import math
import os
import time
from datetime import timedelta
from zipfile import ZipFile

import numpy as np
import slider

empty_beatmap = '''
osu file format v14

[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Soft
StackLeniency: 0
Mode: 0
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 1.0
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1

[Metadata]
Title:audio
TitleUnicode:audio
Artist:Various Artists
ArtistUnicode:Various Artists
Creator:osu_mapper
Version:Insane
Source:
Tags:
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:5
CircleSize:5
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
//Background and Video events
//Break Periods
//Storyboard Layer 0 (Background)
//Storyboard Layer 1 (Fail)
//Storyboard Layer 2 (Pass)
//Storyboard Layer 3 (Foreground)
//Storyboard Layer 4 (Overlay)
//Storyboard Sound Samples

[TimingPoints]


[HitObjects]
'''

ESSENTIAL_FIELDS = ['beatmap_set_id', 'beatmap_id', 'audio_filename', 'artist', 'title', 'version']
DEFAULT_OSZ_DIR = './resources/gen/osz'
DEFAULT_OSU_DIR = './resources/gen/osu'


def osu_filename(beatmap: slider.Beatmap):
    return '%s - %s (%s) [%s].osu' % (beatmap.artist, beatmap.title, beatmap.creator, beatmap.version)


def osz_filename(beatmap: slider.Beatmap):
    if beatmap.beatmap_set_id == -1:
        return 'beatmap-%s-%s.osz' % (str(hash(time.time())), os.path.splitext(beatmap.audio_filename)[0])
    else:
        return '%s %s - %s.osz' % (beatmap.beatmap_set_id, beatmap.artist, beatmap.title)


def set_meta(beatmap: slider.Beatmap, meta: dict):
    for k, v in meta.items():
        if k in beatmap.__dict__:
            beatmap.__setattr__(k, v)
        else:
            print('unknown meta: %s' % k)


def pack_to_osz(audio_path, osu_path_list, osz_path=None, beatmap_list: list = None):
    if beatmap_list is None:
        beatmap_list = [slider.Beatmap.from_path(osu_path) for osu_path in osu_path_list]
    if osz_path is None:
        osz_path = os.path.join(DEFAULT_OSZ_DIR,
                                osz_filename(beatmap_list[0]))
    with ZipFile(osz_path, 'w') as zf:
        zf.write(audio_path, os.path.basename(audio_path))
        for osu_path, beatmap in zip(osu_path_list, beatmap_list):
            zf.write(osu_path, osu_filename(beatmap))


def _valid_types(include_circle=True,
                 include_slider=True,
                 include_spinner=False,
                 include_holdnote=False):
    return tuple(itertools.compress(
        [slider.beatmap.Circle, slider.beatmap.Slider, slider.beatmap.Spinner, slider.beatmap.HoldNote],
        [include_circle, include_slider, include_spinner, include_holdnote],
    ))


def hit_objects(beatmap: slider.Beatmap,
                include_circle=True,
                include_slider=True,
                include_spinner=False,
                include_holdnote=False):
    """
    Returns filterfalse object
    """
    valid_types = _valid_types(include_circle, include_slider, include_spinner, include_holdnote)
    return itertools.filterfalse(
        lambda ho: not isinstance(ho, valid_types),
        beatmap._hit_objects
    )


def get_first_hit_object_time_microseconds(beatmap: slider.Beatmap,
                                           include_circle=True,
                                           include_slider=True,
                                           include_spinner=False,
                                           include_holdnote=False):
    """
    The time offset of the first hit object, in microsecond
    """
    valid_types = _valid_types(include_circle, include_slider, include_spinner, include_holdnote)
    for ho in beatmap._hit_objects:
        if isinstance(ho, valid_types):
            return ho.time / timedelta(microseconds=1)
    print('no hitobject found!')
    return None


def get_first_hit_object_time_milliseconds(beatmap: slider.Beatmap,
                                           include_circle=True,
                                           include_slider=True,
                                           include_spinner=False,
                                           include_holdnote=False):
    """
    The time offset of the first hit object, in microsecond
    """
    valid_types = _valid_types(include_circle, include_slider, include_spinner, include_holdnote)
    for ho in beatmap._hit_objects:
        if isinstance(ho, valid_types):
            return ho.time / timedelta(milliseconds=1)
    print('no hitobject found!')
    return None


def get_conscious_start_time_microseconds(beatmap: slider.Beatmap,
                                          include_circle=True,
                                          include_slider=True,
                                          include_spinner=False,
                                          include_holdnote=False):
    """
    The time offset of min(the first uninherited timing point, the first hit object), in microsecond
    """
    first_obj_time = get_first_hit_object_time_microseconds(
        beatmap, include_circle, include_slider, include_spinner, include_holdnote
    )
    for timing_point in beatmap.timing_points:
        if timing_point.bpm is not None:
            first_timing_point = timing_point.offset / timedelta(microseconds=1)
            return min(first_obj_time, first_timing_point)
    print('no uninherited timing point found, beatmap corrupted?')
    return first_obj_time


def get_conscious_start_time_milliseconds(beatmap: slider.Beatmap,
                                          include_circle=True,
                                          include_slider=True,
                                          include_spinner=False,
                                          include_holdnote=False):
    """
    The time offset of min(the first uninherited timing point, the first hit object), in microsecond
    """
    first_obj_time = get_first_hit_object_time_microseconds(
        beatmap, include_circle, include_slider, include_spinner, include_holdnote
    )
    for timing_point in beatmap.timing_points:
        if timing_point.bpm is not None:
            first_timing_point = timing_point.offset / timedelta(milliseconds=1)
            return min(first_obj_time, first_timing_point)
    print('no uninherited timing point found, beatmap corrupted?')
    return first_obj_time


def get_first_uninherited_timing_point(beatmap: slider.Beatmap):
    """
    The first uninherited timing point.
    """
    for timing_point in beatmap.timing_points:
        if timing_point.parent is None:
            return timing_point
    print('no uninherited timing point found, beatmap corrupted?')
    return None


def get_last_hit_object_time_microseconds(beatmap: slider.Beatmap,
                                          include_circle=True,
                                          include_slider=True,
                                          include_spinner=False,
                                          include_holdnote=False):
    """
    The end_time/time offset of the last hit object, in microsecond
    """
    valid_types = _valid_types(include_circle, include_slider, include_spinner, include_holdnote)
    last_obj = None
    for last_obj in reversed(beatmap._hit_objects):
        if isinstance(last_obj, valid_types):
            break
    if last_obj is None:
        print('no hitobject found!')
        return None
    if isinstance(last_obj, slider.beatmap.Slider):
        return last_obj.end_time / timedelta(microseconds=1)
    elif isinstance(last_obj, slider.beatmap.Spinner):
        return last_obj.end_time / timedelta(microseconds=1)
    else:
        # Circle
        # https://osu.ppy.sh/wiki/zh/Client/File_formats/Osu_%28file_format%29
        return last_obj.time / timedelta(microseconds=1)


def get_last_hit_object_time_milliseconds(beatmap: slider.Beatmap,
                                          include_circle=True,
                                          include_slider=True,
                                          include_spinner=False,
                                          include_holdnote=False):
    """
    The end_time/time offset of the last hit object, in microsecond
    """
    valid_types = _valid_types(include_circle, include_slider, include_spinner, include_holdnote)
    last_obj = None
    for last_obj in reversed(beatmap._hit_objects):
        if isinstance(last_obj, valid_types):
            break
    if last_obj is None:
        print('no hitobject found!')
        return None
    if isinstance(last_obj, slider.beatmap.Slider):
        return last_obj.end_time / timedelta(milliseconds=1)
    elif isinstance(last_obj, slider.beatmap.Spinner):
        return last_obj.end_time / timedelta(milliseconds=1)
    else:
        # Circle
        # https://osu.ppy.sh/wiki/zh/Client/File_formats/Osu_%28file_format%29
        return last_obj.time / timedelta(milliseconds=1)


def get_empty_beatmap():
    return slider.Beatmap.parse(empty_beatmap)


def set_bpm(beatmap: slider.Beatmap, bpm, snap_divisor=8):
    """
    Adds an non inherited timing point with bpm and snaps per beat as specified.
    Also sets BeatDivisor field to snap_divisor
    """
    beatmap.beat_divisor = snap_divisor
    beatmap.timing_points.append(
        slider.beatmap.TimingPoint(
            timedelta(milliseconds=0),
            60000 / bpm,
            snap_divisor,
            0,
            0,
            0,
            None,
            False,
        )
    )


def set_start_time(beatmap: slider.Beatmap, start_time: timedelta):
    """
    Simply by adding a Circle hitobject at start_time.
    """
    beatmap._hit_objects.append(
        slider.beatmap.Circle(
            slider.Position(0, 0),
            start_time,
            0
        )
    )


def set_end_time(beatmap: slider.Beatmap, end_time: timedelta):
    """
    Simply by adding a Circle hitobject at end_time.
    """
    beatmap._hit_objects.append(
        slider.beatmap.Circle(
            slider.Position(0, 0),
            end_time,
            0
        )
    )


def get_snap_per_microseconds(beatmap: slider.Beatmap, snap_divisor=8):
    bpm = beatmap.bpm_min()
    return bpm * snap_divisor / 60000000


def get_snap_microseconds(beatmap: slider.Beatmap, snap_divisor=8):
    bpm = beatmap.bpm_min()
    return 60000000 / bpm / snap_divisor


def get_snap_milliseconds(beatmap: slider.Beatmap, snap_divisor=8):
    bpm = beatmap.bpm_min()
    return 60000 / bpm / snap_divisor


def get_snap_divisor_by_snap_ms(beatmap: slider.Beatmap, snap_ms):
    bpm = beatmap.bpm_min()
    return round(60000 / bpm / snap_ms)


def get_total_snaps(beatmap: slider.Beatmap, snap_divisor=8,
                    include_circle=True,
                    include_slider=True,
                    include_spinner=False,
                    include_holdnote=False):
    snap_ms = get_snap_milliseconds(beatmap, snap_divisor)
    first_ho_time = get_first_hit_object_time_milliseconds(
        beatmap,
        include_circle,
        include_slider,
        include_spinner,
        include_holdnote
    )
    last_ho_time = get_last_hit_object_time_milliseconds(
        beatmap,
        include_circle,
        include_slider,
        include_spinner,
        include_holdnote
    )
    total_snaps = (last_ho_time - first_ho_time) / snap_ms
    # include last snap
    total_snaps = round(total_snaps)
    return total_snaps + 1


def get_difficulty(beatmap: slider.Beatmap):
    try:
        speed_stars = beatmap.speed_stars()
    except ZeroDivisionError:
        # slider may encounter ZeroDivisionError during star calculation
        # speed_stars have the closest relationship with hit_object determination
        # if speed_stars are unavailable, try overall stars or overall difficulty(this is always available)
        # higher the speed_stars, the more likely a hit_object exists at that snap
        try:
            speed_stars = beatmap.stars()
        except ZeroDivisionError:
            speed_stars = beatmap.overall_difficulty
    return speed_stars


def single_non_inherited_timepoints(beatmap: slider.Beatmap):
    non_inherited_timepoint_num = 0
    for ti, timing_point in enumerate(beatmap.timing_points):
        if timing_point.bpm is not None:
            non_inherited_timepoint_num += 1
    return non_inherited_timepoint_num == 1


def check_essential_fields(beatmap: slider.Beatmap, fields=ESSENTIAL_FIELDS):
    """
    Check if essential fields exist in slider.Beatmap
    """
    for field in fields:
        if getattr(beatmap, field) is None:
            print('essential field %s is None!' % field)
            return False
    return True


def slider_snap_pos(ho: slider.beatmap.Slider, snap_num):
    """
    include start pos
    """
    total_length = ho.length * ho.repeat
    snap_accumulate_length = np.arange(0, total_length, total_length / snap_num)
    snap_accumulate_length = np.append(snap_accumulate_length, total_length)
    all_pos = []
    for i, cur_length in enumerate(snap_accumulate_length):
        direction, remain = cur_length // ho.length, (cur_length % ho.length) / ho.length
        # direction = 0: forward, 1: backward
        if direction % 2 == 1:
            remain = 1 - remain
        all_pos.append(ho.curve(remain))
    return all_pos


def add_circle(beatmap, pos, time, hitsound=0, addition='0:0:0:0:'):
    """
    time can be number (in milliseconds), or timedelta
    """
    if not isinstance(time, timedelta):
        time = timedelta(milliseconds=time)
    ho_circle = slider.beatmap.Circle(
        slider.Position(pos[0], pos[1]),
        time,
        hitsound,
        addition
    )
    beatmap._hit_objects.append(ho_circle)
    return ho_circle


def add_spinner(beatmap, pos, time, num_beats, ms_per_beat, hitsound=0, addition='0:0:0:0:'):
    """
    time can be number (in milliseconds), or timedelta
    """
    if not isinstance(time, timedelta):
        time = timedelta(milliseconds=time)
    duration = timedelta(milliseconds=int(num_beats * ms_per_beat))
    ho_spinner = slider.beatmap.Spinner(
        slider.Position(pos[0], pos[1]),
        time,
        hitsound,
        time + duration,
        addition
    )
    beatmap._hit_objects.append(ho_spinner)
    return ho_spinner


def add_slider(beatmap,
               curve_type,
               pos_list,
               time,
               num_beats,
               ms_per_beat,
               repeat=1,
               hitsound=0,
               edge_sounds=[0, 0],
               edge_sets=['0:0', '0:0'],
               addition='0:0:0:0:',
               slider_multiplier=1,
               slider_tick_rate=0.1,
               ):
    """
    time can be number (in milliseconds), or timedelta
    ms_per_beat : float
        The milliseconds per beat, this is another representation of BPM.
    """
    # velocity_multiplier = -100 / ms_per_beat
    pos_list = [slider.Position(x, y) for x, y in pos_list]
    if not isinstance(time, timedelta):
        time = timedelta(milliseconds=time)
    # pixels_per_beat = slider_multiplier * 100 * velocity_multiplier
    pixels_per_beat = slider_multiplier * 100
    pixel_length = pixels_per_beat * num_beats / repeat
    ticks = int((math.ceil((num_beats - 0.1) / repeat * slider_tick_rate) - 1) * repeat + repeat + 1)
    duration = timedelta(milliseconds=int(num_beats * ms_per_beat))
    # print(pos_list)
    ho_slider = slider.beatmap.Slider(
        pos_list[0],
        time,
        time + duration,
        hitsound,
        slider.beatmap.Curve.from_kind_and_points(
            curve_type, pos_list, pixel_length
        ),
        repeat,
        pixel_length,
        ticks,
        num_beats,
        slider_tick_rate,
        ms_per_beat,
        edge_sounds,
        edge_sets,
        addition
    )
    beatmap._hit_objects.append(ho_slider)
    return ho_slider


class BeatmapConstructor:
    def __init__(self, beatmap: slider.Beatmap):
        self.beatmap = beatmap
        self.ms_per_beat = 60000 / self.beatmap.bpm_min()
        self.slider_multiplier = self.beatmap.slider_multiplier
        self.slider_tick_rate = self.beatmap.slider_tick_rate

    def add_circle(self, pos, time, hitsound=0, addition='0:0:0:0:'):
        return add_circle(self.beatmap, pos, time, hitsound, addition)

    def add_spinner(self, pos, time, num_beats, hitsound=0, addition='0:0:0:0:'):
        return add_spinner(self.beatmap, pos, time, num_beats, self.ms_per_beat, hitsound, addition)

    def add_slider(self,
                   curve_type,
                   pos_list,
                   time,
                   num_beats,
                   repeat=1,
                   hitsound=0,
                   edge_sounds=[0, 0],
                   edge_sets=['0:0', '0:0'],
                   addition='0:0:0:0:'):
        return add_slider(
            self.beatmap,
            curve_type,
            pos_list,
            time,
            num_beats,
            self.ms_per_beat,
            repeat,
            hitsound,
            edge_sounds,
            edge_sets,
            addition,
            self.slider_multiplier,
            self.slider_tick_rate,
        )
