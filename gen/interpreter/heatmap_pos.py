from datetime import timedelta
from itertools import chain
import numpy as np
import slider

from gen.position_generator import RandomWalkInRectangle
from preprocess.dataset.heatmap_datasetv2 import time_to_frame
from util import beatmap_util
import random


class HeatmapPosInterpreter:
    def __init__(self, sr, hop_length, n_fft):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft

    @staticmethod
    def map_pos(x, y):
        if x < 0:
            x = 0.1
        if x > 1:
            x = 0.9
        if y < 0:
            y = 0.1
        if y > 1:
            y = 0.9
        x = int(x * (691 + 180) - 180)
        y = int(y * (407 + 82) - 82)
        return x, y

    @staticmethod
    def parse_step_label(step_label):
        circle_density, slider_density, _, x, y = step_label
        x, y = HeatmapPosInterpreter.map_pos(x, y)
        if circle_density < 0.5 and slider_density < 0.5:
            return 0, x, y
        if circle_density > slider_density:
            return 1, x, y
        else:
            return 2, x, y

    def gen_hitobjects(self, beatmap: slider.Beatmap, labels, bpm, start_ms=None, end_ms=None, snap_divisor=8):
        """
        labels:     circle hit, slider hit, spinner hit, cursor x, cursor y
        """
        ms_per_beat = 60000 / bpm
        snap_ms = ms_per_beat / snap_divisor
        circle_hit, slider_hit, spinner_hit, cursor_x, cursor_y = labels.T
        total_len = labels.shape[0]
        circle_peaks = np.where(circle_hit < 0.5, 0, circle_hit)
        padded = np.pad(circle_peaks, (1, 1))
        # local peaks
        circle_peaks = np.where((circle_peaks > padded[:-2]) & (circle_peaks > padded[2:]), 1, 0)
        peak_occupied = np.zeros(circle_peaks.shape)

        def parse_at_time():
            x, y = HeatmapPosInterpreter.map_pos(cursor_x[frame], cursor_y[frame])
            if slider_hit[frame] > 0.5:
                return 2, x, y
            if peak_occupied[frame] == 0 and circle_peaks[frame] > 0:
                peak_occupied[frame] = 1
                return 1, x, y
            return 0, x, y

        snap_idx = 0
        while True:
            time = start_ms + snap_idx * snap_ms
            frame = time_to_frame(time, self.sr, self.hop_length, self.n_fft)
            if end_ms is not None:
                if time > end_ms:
                    break
            label, x, y = parse_at_time()
            if label == 1:
                # circles, period == 1
                beatmap_util.add_circle(beatmap, (x, y), time)
                snap_idx += 1
                # print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
            elif label == 2:
                pos_list = [[x, y]]
                start_pos = snap_idx
                # slider start
                while snap_idx < len(labels):
                    label, x, y = parse_at_time()
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
                # print(('add slider at (%.3f, %s)' % (time, ' ({}, {})' * len(pos_list))).format(*chain(*pos_list)))
            else:
                snap_idx += 1
