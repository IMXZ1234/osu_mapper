import os
from datetime import timedelta
import numpy as np

import slider
from slider import Position

from util import beatmap_util

NOHO_LABEL = 0
CIRCLE_LABEL = 1
SLIDER_LABEL = 2
SPINNER_LABEL = 3
HOLDNOTE_LABEL = 4

KEEP_STATE_LABEL = 0
SWITCH_NOHO_LABEL = 1
SWITCH_CIRCLE_LABEL = 2
SWITCH_SLIDER_LABEL = 3
SWITCH_SPINNER_LABEL = 4
SWITCH_HOLDNOTE_LABEL = 5


def hitobjects_to_label(beatmap: slider.Beatmap, aligned_start_time, snap_per_microsecond, total_snap_num,
                        align_to_snaps=None, multi_label=False,
                        include_circle=True, include_slider=True, include_spinner=False, include_holdnote=False):
    """
    if multi_label:
    0: no hit object
    1: circle
    2: slider
    3: spinner
    4: hold note

    if not multi_label:
    0: no hit object
    1: has hit object
    """
    # print(audio_start_time_offset)
    if align_to_snaps is not None:
        if total_snap_num % align_to_snaps != 0:
            total_snap_num = (total_snap_num // align_to_snaps + 1) * align_to_snaps
    label = [0 for _ in range(total_snap_num)]
    for ho in beatmap_util.hit_objects(beatmap,
                                       include_circle,
                                       include_slider,
                                       include_spinner,
                                       include_holdnote):
        snap_idx = (ho.time / timedelta(microseconds=1) - aligned_start_time) * snap_per_microsecond
        # if abs(round(snap_idx) - snap_idx) > 0.2:
        #     print('snap_idx error to large!')
        #     print(snap_idx)
        snap_idx = round(snap_idx)
        if snap_idx >= total_snap_num:
            print('snap index %d out of bound! total snap num %d' % (snap_idx, total_snap_num))
            snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Circle):
            label[snap_idx] = CIRCLE_LABEL
            continue
        end_snap_idx = (ho.end_time / timedelta(microseconds=1) - aligned_start_time) * snap_per_microsecond
        if end_snap_idx >= total_snap_num:
            print('end snap index %d out of bound! total snap num %d' % (end_snap_idx, total_snap_num))
            end_snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Slider):
            label_value = SLIDER_LABEL
        elif isinstance(ho, slider.beatmap.Spinner):
            label_value = SPINNER_LABEL
        else:
            # HoldNote
            label_value = HOLDNOTE_LABEL
        if not multi_label:
            label_value = CIRCLE_LABEL
        # if abs(round(end_snap_idx) - end_snap_idx) > 0.2:
        #     print('end_snap_idx error to large!')
        #     print(end_snap_idx)
        # print('before round end_snap_idx')
        # print(end_snap_idx)
        end_snap_idx = round(end_snap_idx)
        # print('end_snap_idx')
        # print(end_snap_idx)
        for i in range(snap_idx, end_snap_idx + 1):
            label[i] = label_value
    return label


def hitobjects_to_label_v2(beatmap: slider.Beatmap, aligned_ms=None, snap_ms=None, total_snap_num=None,
                           align_to_snaps=None, multi_label=False, multibeat_label_fmt=0,
                           include_circle=True, include_slider=True, include_spinner=False, include_holdnote=False):
    """
    if multi_label:
    0: no hit object
    1: circle
    2: slider
    3: spinner
    4: hold note

    if not multi_label:
    0: no hit object
    1: has hit object
    """
    if snap_ms is None:
        # if not specified, we calculate using original snap_divisor
        snap_ms = beatmap_util.get_snap_milliseconds(beatmap, beatmap.beat_divisor)
    snap_divisor = beatmap_util.get_snap_divisor_by_snap_ms(beatmap, snap_ms)
    ho_base_itv = [
        0,
        1,
        snap_divisor,  # sliders will always last for more than one beat, and covers at least 2 beats
        snap_divisor,  # spinners will always last for more than one beat, and covers at least 2 beats
        snap_divisor,  # holdnotes will always last for more than one beat, and covers at least 2 beats
    ]
    if aligned_ms is None:
        aligned_ms = beatmap_util.get_first_hit_object_time_milliseconds(
            beatmap,
            include_circle,
            include_slider,
            include_spinner,
            include_holdnote
        )
    if total_snap_num is None:
        total_snap_num = round((beatmap_util.get_last_hit_object_time_milliseconds(
            beatmap,
            include_circle,
            include_slider,
            include_spinner,
            include_holdnote
        ) - aligned_ms) / snap_ms) + 1
    # print(audio_start_time_offset)
    if align_to_snaps is not None:
        if total_snap_num % align_to_snaps != 0:
            total_snap_num = (total_snap_num // align_to_snaps + 1) * align_to_snaps
    label = [0 for _ in range(total_snap_num)]
    for ho in beatmap_util.hit_objects(beatmap,
                                       include_circle,
                                       include_slider,
                                       include_spinner,
                                       include_holdnote):
        snap_idx = (ho.time / timedelta(milliseconds=1) - aligned_ms) / snap_ms
        if abs(round(snap_idx) - snap_idx) > 0.1:
            print('snap_idx error too large!')
            print(snap_idx)
        snap_idx = round(snap_idx)
        if snap_idx >= total_snap_num:
            print('snap index %d out of bound! total snap num %d' % (snap_idx, total_snap_num))
            snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Circle):
            label[snap_idx] = CIRCLE_LABEL
            continue
        end_snap_idx = (ho.end_time / timedelta(milliseconds=1) - aligned_ms) / snap_ms
        if end_snap_idx >= total_snap_num:
            print('end snap index %d out of bound! total snap num %d' % (end_snap_idx, total_snap_num))
            end_snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Slider):
            label_value = SLIDER_LABEL
        elif isinstance(ho, slider.beatmap.Spinner):
            label_value = SPINNER_LABEL
        else:
            # HoldNote
            label_value = HOLDNOTE_LABEL
        if not multi_label:
            label_value = CIRCLE_LABEL
        # if abs(round(end_snap_idx) - end_snap_idx) > 0.2:
        #     print('end_snap_idx error to large!')
        #     print(end_snap_idx)
        # print('before round end_snap_idx')
        # print(end_snap_idx)
        end_snap_idx = round(end_snap_idx)
        # print('end_snap_idx')
        # print(end_snap_idx)
        if multibeat_label_fmt == 0:
            for i in range(snap_idx, end_snap_idx + 1):
                label[i] = label_value
        else:
            # only set label at beginning of beats
            for i in range(snap_idx, end_snap_idx + 1, ho_base_itv[label_value]):
                label[i] = label_value
    return label


def hitobjects_to_label_with_pos(beatmap: slider.Beatmap, aligned_ms=None, snap_ms=None, total_snap_num=None,
                           align_to_snaps=None, multi_label=False, multibeat_label_fmt=0,
                           include_circle=True, include_slider=True, include_spinner=False, include_holdnote=False):
    """
    if multi_label:
    0: no hit object
    1: circle
    2: slider
    3: spinner
    4: hold note

    if not multi_label:
    0: no hit object
    1: has hit object
    """
    if snap_ms is None:
        # if not specified, we calculate using original snap_divisor
        snap_ms = beatmap_util.get_snap_milliseconds(beatmap, beatmap.beat_divisor)
    snap_divisor = beatmap_util.get_snap_divisor_by_snap_ms(beatmap, snap_ms)
    ho_base_itv = [
        0,
        1,
        snap_divisor,  # sliders will always last for more than one beat, and covers at least 2 beats
        snap_divisor,  # spinners will always last for more than one beat, and covers at least 2 beats
        snap_divisor,  # holdnotes will always last for more than one beat, and covers at least 2 beats
    ]
    if aligned_ms is None:
        aligned_ms = beatmap_util.get_first_hit_object_time_milliseconds(
            beatmap,
            include_circle,
            include_slider,
            include_spinner,
            include_holdnote
        )
    if total_snap_num is None:
        total_snap_num = round((beatmap_util.get_last_hit_object_time_milliseconds(
            beatmap,
            include_circle,
            include_slider,
            include_spinner,
            include_holdnote
        ) - aligned_ms) / snap_ms) + 1
    # print(audio_start_time_offset)
    if align_to_snaps is not None:
        if total_snap_num % align_to_snaps != 0:
            total_snap_num = (total_snap_num // align_to_snaps + 1) * align_to_snaps
    label = np.zeros([total_snap_num, 3], dtype=float)
    last_end_pos = None
    last_end_snap_idx = None
    for ho in beatmap_util.hit_objects(beatmap,
                                       include_circle,
                                       include_slider,
                                       include_spinner,
                                       include_holdnote):
        start_pos = ho.position
        snap_idx = (ho.time / timedelta(milliseconds=1) - aligned_ms) / snap_ms
        if abs(round(snap_idx) - snap_idx) > 0.1:
            print('snap_idx error too large!')
            print(snap_idx)
        snap_idx = round(snap_idx)
        if last_end_pos is not None:
            # linear interpolate between two object to simulate cursor trajectory during the interval
            for i in range(last_end_snap_idx + 1, snap_idx):
                start_pos_array = np.array([start_pos.x, start_pos.y])
                last_end_pos_array = np.array([last_end_pos.x, last_end_pos.y])
                snap_pos_diff = (last_end_pos_array - start_pos_array) / (snap_idx - last_end_snap_idx)
                label[i, 1:] = snap_pos_diff * (i - last_end_snap_idx) + last_end_pos_array
        if snap_idx >= total_snap_num:
            print('snap index %d out of bound! total snap num %d' % (snap_idx, total_snap_num))
            snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Circle):
            label[snap_idx] = np.array([CIRCLE_LABEL, start_pos.x, start_pos.y])
            last_end_pos = ho.position
            last_end_snap_idx = snap_idx
            continue
        end_snap_idx = (ho.end_time / timedelta(milliseconds=1) - aligned_ms) / snap_ms
        if end_snap_idx >= total_snap_num:
            print('end snap index %d out of bound! total snap num %d' % (end_snap_idx, total_snap_num))
            end_snap_idx = total_snap_num - 1

        end_snap_idx = round(end_snap_idx)

        if isinstance(ho, slider.beatmap.Slider):
            label_value = SLIDER_LABEL
            pos_list = beatmap_util.slider_snap_pos(ho, end_snap_idx - snap_idx)
        elif isinstance(ho, slider.beatmap.Spinner):
            label_value = SPINNER_LABEL
            pos_list = [start_pos for _ in range(snap_idx, end_snap_idx + 1)]
        else:
            # HoldNote
            label_value = HOLDNOTE_LABEL
            pos_list = [start_pos for _ in range(snap_idx, end_snap_idx + 1)]
        if len(pos_list) != end_snap_idx - snap_idx + 1:
            print('pos list len check failed')
            return None
        if not multi_label:
            label_value = CIRCLE_LABEL
        # if abs(round(end_snap_idx) - end_snap_idx) > 0.2:
        #     print('end_snap_idx error to large!')
        #     print(end_snap_idx)
        # print('before round end_snap_idx')
        # print(end_snap_idx)
        # print('end_snap_idx')
        # print(end_snap_idx)
        if multibeat_label_fmt == 0:
            for i in range(snap_idx, end_snap_idx + 1):
                pos = pos_list[i - snap_idx]
                label[i] = np.array([label_value, pos.x, pos.y])
                # if pos.x < 0 or pos.y < 0:
                #     print(type(pos))
        else:
            # only set label at beginning of beats
            for i in range(snap_idx, end_snap_idx + 1, ho_base_itv[label_value]):
                pos = pos_list[i - snap_idx]
                label[i] = np.array([label_value, pos.x, pos.y])
        last_end_pos = pos_list[-1]
        last_end_snap_idx = snap_idx
    # only hit_objects within x:[-180, 691], y:[-82, 407] are visible, for beatmaps generated using osu! beatmap editor
    # thus we assume at any time cursor will not get out of this rectangle
    # label[:, 1:] = (label[:, 1:] + [180, 82]) / np.array([691 + 180, 407 + 82])
    # if (label[:, 1:] > 1.1).any():
    #     print('> 1')
    # if (label[:, 1:] < -0.1).any():
    #     print('< 0')
    min_x, min_y = np.min(label[:, 1:], axis=0)
    max_x, max_y = np.max(label[:, 1:], axis=0)
    # print((min_x, max_x))
    # print((min_y, max_y))
    if min_x == max_x:
        print(beatmap.title)
        print(beatmap.hit_objects())
        print(len(beatmap.hit_objects()))
        print(label)
        print(label.shape)
        beatmap.write_path(
            os.path.join(r'')
        )

    return label


def hitobjects_to_label_switch(beatmap: slider.Beatmap, aligned_ms, snap_ms, total_snap_num,
                               align_to_snaps=None, multi_label=False,
                               include_circle=True, include_slider=True, include_spinner=False, include_holdnote=False):
    """
    if multi_label:
    0: keep state
    1: switch to state which outputs nothing
    2: circle
    3: switch to state which outputs slider
    4: switch to state which outputs spinner
    5: hold note

    if not multi_label:
    0: keep state
    1: switch to state which outputs nothing
    2: switch to state which outputs hit objects
    """
    # print(audio_start_time_offset)
    mapping = list(range(6))
    if not multi_label:
        for i in range(3, 6):
            mapping[i] = 2
    if align_to_snaps is not None:
        if total_snap_num % align_to_snaps != 0:
            total_snap_num = (total_snap_num // align_to_snaps + 1) * align_to_snaps
    label = [mapping[KEEP_STATE_LABEL] for _ in range(total_snap_num)]
    for ho in beatmap_util.hit_objects(beatmap,
                                       include_circle,
                                       include_slider,
                                       include_spinner,
                                       include_holdnote):
        snap_idx = (ho.time / timedelta(milliseconds=1) - aligned_ms) / snap_ms
        if abs(round(snap_idx) - snap_idx) > 0.1:
            print('snap_idx error to large!')
            print(snap_idx)
        snap_idx = round(snap_idx)
        if snap_idx >= total_snap_num:
            print('snap index %d out of bound! total snap num %d' % (snap_idx, total_snap_num))
            snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Circle):
            label[snap_idx] = mapping[SWITCH_CIRCLE_LABEL]
            continue
        elif isinstance(ho, slider.beatmap.Slider):
            label[snap_idx] = mapping[SWITCH_SLIDER_LABEL]
        elif isinstance(ho, slider.beatmap.Spinner):
            label[snap_idx] = mapping[SWITCH_SPINNER_LABEL]
        elif isinstance(ho, slider.beatmap.HoldNote):
            label[snap_idx] = mapping[SWITCH_HOLDNOTE_LABEL]

        end_snap_idx = (ho.end_time / timedelta(milliseconds=1) - aligned_ms) / snap_ms
        if end_snap_idx >= total_snap_num:
            print('end snap index %d out of bound! total snap num %d' % (end_snap_idx, total_snap_num))
            end_snap_idx = total_snap_num - 1
        end_snap_idx = round(end_snap_idx)
        label[end_snap_idx] = mapping[SWITCH_NOHO_LABEL]
    return label


def cls_label_to_density(cls_label):
    # print('cls_label')
    # print(len(cls_label))
    circle_kernel = np.array([0.2, 0.5, 1, 0.5, 0.2])
    slider_kernel = np.array([0, 0.2, 0.5, 0.2, 0])
    kernel_half_len = len(circle_kernel) // 2
    density = np.zeros(len(cls_label) + len(circle_kernel) - 1)
    for i, label in enumerate(cls_label):
        if cls_label == 0:
            continue
        if cls_label == 1:
            kernel = circle_kernel
        else:
            kernel = slider_kernel
        density[i:i + 2 * kernel_half_len + 1] += kernel
    density_label = density[kernel_half_len:len(cls_label) + kernel_half_len]
    density_label = np.clip(density_label, a_min=None, a_max=1)
    # print('density_label')
    # print(len(density_label))
    # assert density_label == cls_label
    return density_label


def hitobjects_to_density(beatmap: slider.Beatmap, aligned_start_time, snap_per_microsecond, total_snap_num,
                          align_to_snaps=None,
                          include_circle=True, include_slider=True, include_spinner=False, include_holdnote=False):
    cls_label = hitobjects_to_label(beatmap, aligned_start_time, snap_per_microsecond, total_snap_num,
                                    align_to_snaps, include_circle, include_slider, include_spinner, include_holdnote)
    return cls_label_to_density(cls_label)


def hitobjects_to_density_v2(beatmap: slider.Beatmap, aligned_ms, snap_ms, total_snap_num,
                             align_to_snaps=None, multi_label=False,
                             include_circle=True, include_slider=True, include_spinner=False, include_holdnote=False):
    cls_label = hitobjects_to_label_v2(beatmap, aligned_ms, snap_ms, total_snap_num,
                                       align_to_snaps, multi_label,
                                       include_circle, include_slider, include_spinner, include_holdnote)
    return cls_label_to_density(cls_label)


def hitobjects_to_label_type_specific(beatmap: slider.Beatmap, aligned_ms=None, snap_ms=None, total_snap_num=None,
                           align_to_snaps=None, multi_label=False,
                           include_circle=True, include_slider=True, include_spinner=False, include_holdnote=False):
    """
    if multi_label:
    0: no hit object
    1: circle
    2: slider
    3: spinner
    4: hold note

    if not multi_label:
    0: no hit object
    1: has hit object
    """
    if aligned_ms is None:
        aligned_ms = beatmap_util.get_first_hit_object_time_milliseconds(
            beatmap,
            include_circle,
            include_slider,
            include_spinner,
            include_holdnote
        )
    if snap_ms is None:
        # if not specified, we calculate using original snap_divisor
        snap_ms = beatmap_util.get_snap_milliseconds(beatmap, beatmap.beat_divisor)
    if total_snap_num is None:
        total_snap_num = round((beatmap_util.get_last_hit_object_time_milliseconds(
            beatmap,
            include_circle,
            include_slider,
            include_spinner,
            include_holdnote
        ) - aligned_ms) / snap_ms) + 1
    # print(audio_start_time_offset)
    if align_to_snaps is not None:
        if total_snap_num % align_to_snaps != 0:
            total_snap_num = (total_snap_num // align_to_snaps + 1) * align_to_snaps
    label = [0 for _ in range(total_snap_num)]
    for ho in beatmap_util.hit_objects(beatmap,
                                       include_circle,
                                       include_slider,
                                       include_spinner,
                                       include_holdnote):
        snap_idx = (ho.time / timedelta(milliseconds=1) - aligned_ms) / snap_ms
        if abs(round(snap_idx) - snap_idx) > 0.1:
            print('snap_idx error to large!')
            print(snap_idx)
        snap_idx = round(snap_idx)
        if snap_idx >= total_snap_num:
            print('snap index %d out of bound! total snap num %d' % (snap_idx, total_snap_num))
            snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Circle):
            label[snap_idx] = CIRCLE_LABEL
            continue
        end_snap_idx = (ho.end_time / timedelta(milliseconds=1) - aligned_ms) / snap_ms
        if end_snap_idx >= total_snap_num:
            print('end snap index %d out of bound! total snap num %d' % (end_snap_idx, total_snap_num))
            end_snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Slider):
            label_value = SLIDER_LABEL
        elif isinstance(ho, slider.beatmap.Spinner):
            label_value = SPINNER_LABEL
        else:
            # HoldNote
            label_value = HOLDNOTE_LABEL
        if not multi_label:
            label_value = CIRCLE_LABEL
        # if abs(round(end_snap_idx) - end_snap_idx) > 0.2:
        #     print('end_snap_idx error to large!')
        #     print(end_snap_idx)
        # print('before round end_snap_idx')
        # print(end_snap_idx)
        end_snap_idx = round(end_snap_idx)
        # print('end_snap_idx')
        # print(end_snap_idx)
        for i in range(snap_idx, end_snap_idx + 1):
            label[i] = label_value
    return label