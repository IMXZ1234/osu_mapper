from datetime import timedelta
import numpy as np

import slider


from util import beatmap_util


CIRCLE_LABEL = 1
SLIDER_LABEL = 2
SPINNER_LABEL = 3
HOLDNOTE_LABEL = 1


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


def hitobjects_to_label_v2(beatmap: slider.Beatmap, aligned_ms, snap_ms, total_snap_num,
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


def hitobjects_to_density(beatmap: slider.Beatmap, aligned_start_time, snap_per_microsecond, total_snap_num,
                         align_to_snaps=None,
                         include_circle=True, include_slider=True, include_spinner=False, include_holdnote=False):
    cls_label = hitobjects_to_label(beatmap, aligned_start_time, snap_per_microsecond, total_snap_num,
                                    align_to_snaps, include_circle, include_slider, include_spinner, include_holdnote)
    # print('cls_label')
    # print(len(cls_label))
    kernel = np.array([0.2, 0.5, 1, 0.5, 0.2])
    kernel_half_len = len(kernel) // 2
    density = np.zeros(len(cls_label) + len(kernel) - 1)
    should_add_kernel = [False, include_circle, include_slider, include_spinner, include_holdnote]
    for i in range(len(cls_label)):
        if should_add_kernel[cls_label[i]]:
            density[i:i+2*kernel_half_len+1] += kernel
    density_label = density[kernel_half_len:len(cls_label)+kernel_half_len]
    density_label = np.clip(density_label, a_min=None, a_max=1)
    # print('density_label')
    # print(len(density_label))
    # assert density_label == cls_label
    return density_label
