from itertools import chain

import slider

from postprocess import embedding_decode
from util import beatmap_util


class CursorWithHitEmbeddingInterpreter:
    def __init__(self, embedding_path, beat_divisor, subset_path):
        self.beat_divisor = beat_divisor
        self.decoder = embedding_decode.EmbeddingDecode(embedding_path, beat_divisor, subset_path)

    @staticmethod
    def map_pos(x, y):
        # x = int(x * (691 + 180) - 180)
        # y = int(y * (407 + 82) - 82)
        x = int(x * 512 + 256)
        y = int(y * 384 + 192)
        # if x < 0:
        #     x = 0.1
        # if x > 1:
        #     x = 0.9
        # if y < 0:
        #     y = 0.1
        # if y > 1:
        #     y = 0.9
        return x, y

    def gen_hitobjects(self, beatmap: slider.Beatmap, labels, bpm, start_ms=None, end_ms=None):
        """
        labels: coord_output(n_snaps, 2), embedding_output(n_beats, C)
        """
        coord_output, embedding_output = labels
        print(coord_output.shape)
        print(coord_output)
        print(embedding_output.shape)
        print(embedding_output)
        # into snap-based hit_signal with same length as coord_output
        hit_signal = self.decoder.decode(embedding_output)
        print(hit_signal.shape)
        print(hit_signal.tolist())
        assert len(hit_signal) == len(coord_output)
        ms_per_beat = 60000 / bpm
        snap_ms = ms_per_beat / self.beat_divisor
        cursor_x, cursor_y = coord_output.T

        snap_idx = 0
        while snap_idx < len(hit_signal):
            time = start_ms + snap_idx * snap_ms
            if end_ms is not None:
                if time > end_ms:
                    break
            x, y = CursorWithHitEmbeddingInterpreter.map_pos(cursor_x[snap_idx], cursor_y[snap_idx])
            label = hit_signal[snap_idx]
            if label == 1:
                # circles, period == 1
                beatmap_util.add_circle(beatmap, (x, y), time)
                snap_idx += 1
                print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
            elif label == 2:
                pos_list = [[x, y]]
                start_pos = snap_idx
                # slider start
                while snap_idx < len(hit_signal):
                    x, y = CursorWithHitEmbeddingInterpreter.map_pos(cursor_x[snap_idx], cursor_y[snap_idx])
                    label = hit_signal[snap_idx]
                    if label == 2:
                        pos_list.append([x, y])
                        snap_idx += 1
                    else:
                        break
                if snap_idx - start_pos < 2:
                    # bad slider
                    print('bad slider at %.3f' % time)
                    # beatmap_util.add_circle(beatmap, (x, y), time)
                    continue
                num_beats = (snap_idx-start_pos) / self.beat_divisor

                ho_slider = beatmap_util.add_slider(
                    beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                )
                print(('add slider at (%.3f, %s)' % (time, ' ({}, {})' * len(pos_list))).format(*chain(*pos_list)))
            else:
                snap_idx += 1
