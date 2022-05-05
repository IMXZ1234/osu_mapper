from datetime import timedelta

import slider


from util import beatmap_util


def hit_objects_to_label(beatmap, aligned_start_time, snap_per_microsecond, total_snap_num,
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
            label[snap_idx] = 1
            continue
        end_snap_idx = (ho.end_time / timedelta(microseconds=1) - aligned_start_time) * snap_per_microsecond
        if end_snap_idx >= total_snap_num:
            print('end snap index %d out of bound! total snap num %d' % (end_snap_idx, total_snap_num))
            end_snap_idx = total_snap_num - 1
        if isinstance(ho, slider.beatmap.Slider):
            label_value = 2
        elif isinstance(ho, slider.beatmap.Spinner):
            label_value = 3
        else:
            # HoldNote
            label_value = 4
        if not multi_label:
            label_value = 1
        # if abs(round(end_snap_idx) - end_snap_idx) > 0.2:
        #     print('end_snap_idx error to large!')
        #     print(end_snap_idx)
        end_snap_idx = round(end_snap_idx)
        for i in range(snap_idx, end_snap_idx + 1):
            label[i] = label_value
    return label
