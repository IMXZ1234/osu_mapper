from datetime import timedelta
from itertools import chain
import numpy as np
import slider

from gen.position_generator import RandomWalkInRectangle
from preprocess.dataset.heatmap_datasetv2 import time_to_frame
from util import beatmap_util
import random


class ReMapInterpreter:
    def __init__(self):
        pass

    @staticmethod
    def map_pos(x, y):
        x = int(x * 512 + 256)
        y = int(y * 384 + 192)
        return x, y

    def gen_hitobjects(self, beatmap: slider.Beatmap, labels, bpm, start_ms=None, end_ms=None, snap_divisor=8):
        """
        labels: [n_snaps, 3]
        snap_type, x_pos_seq, y_pos_seq
        """
        ms_per_beat = 60000 / bpm
        snap_ms = ms_per_beat / snap_divisor
        # circle_hit, slider_hit, spinner_hit, cursor_x, cursor_y = labels.T
        total_len = labels.shape[0]
        num_hit_objects = 0
        snap_type, coord = labels[:, 0], labels[:, 1:]

        snap_idx = 0
        while True:
            time = start_ms + snap_idx * snap_ms
            if end_ms is not None:
                if time > end_ms:
                    break
            x, y = self.map_pos(coord[snap_idx][0], coord[snap_idx][1])
            label = snap_type[snap_idx]
            if label == 1:
                # circles, period == 1
                beatmap_util.add_circle(beatmap, (x, y), time)
                snap_idx += 1
                print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
                num_hit_objects += 1
            elif label == 2:
                pos_list = [[x, y]]
                start_pos = snap_idx
                # slider start
                while snap_idx < len(labels):
                    x, y = self.map_pos(coord[snap_idx][0], coord[snap_idx][1])
                    label = snap_type[snap_idx]
                    if label == 2:
                        pos_list.append([x, y])
                        snap_idx += 1
                    else:
                        break
                if snap_idx - start_pos < 2:
                    # bad slider
                    # beatmap_util.add_circle(beatmap, (x, y), time)
                    continue
                num_beats = (snap_idx-start_pos) / snap_divisor

                ho_slider = beatmap_util.add_slider(
                    beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                )
                print('add slider at (%.3f, %s)' % (time, (' ({}, {})' * len(pos_list)).format(*chain(*pos_list))))
                num_hit_objects += 1
            elif label == 3:
                pos_list = [[x, y]]
                start_pos = snap_idx
                # slider start
                while snap_idx < len(labels):
                    x, y = self.map_pos(coord[snap_idx][0], coord[snap_idx][1])
                    label = snap_type[snap_idx]
                    if label == 3:
                        pos_list.append([x, y])
                        snap_idx += 1
                    else:
                        break
                if snap_idx - start_pos < 2:
                    # bad slider
                    # beatmap_util.add_circle(beatmap, (x, y), time)
                    continue
                num_beats = (snap_idx-start_pos) / snap_divisor

                ho_spinner = beatmap_util.add_spinner(
                    beatmap, pos_list[0], time, num_beats, ms_per_beat
                )
                print('add spinner at (%.3f)' % time)
                num_hit_objects += 1
            else:
                snap_idx += 1
            if snap_idx >= total_len:
                break

        print('total %d hitobjects' % num_hit_objects)
