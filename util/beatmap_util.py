from datetime import timedelta
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
Creator:IMXZ123
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

ESSENTIAL_FIELDS = ['beatmap_set_id', 'beatmap_id', 'audio_filename']


def get_first_hit_object_time_microseconds(beatmap: slider.Beatmap):
    """
    The time offset of the first hit object, in microsecond
    """
    # print('beatmap.beatmap_id')
    # print(beatmap.beatmap_id)
    # print('beatmap.beatmap_set_id')
    # print(beatmap.beatmap_set_id)
    first_obj_time = beatmap.hit_objects()[0].time / timedelta(microseconds=1)
    return first_obj_time


def get_conscious_start_time_microseconds(beatmap: slider.Beatmap):
    """
    The time offset of min(the first uninherited timing point, the first hit object), in microsecond
    """
    first_obj_time = beatmap.hit_objects()[0].time / timedelta(microseconds=1)
    for timing_point in beatmap.timing_points:
        if timing_point.bpm is not None:
            first_timing_point = timing_point.offset / timedelta(microseconds=1)
            return min(first_obj_time, first_timing_point)
    print('no uninherited timing point found, beatmap corrupted?')
    return first_obj_time


def get_last_hit_object_time_microseconds(beatmap: slider.Beatmap):
    """
    The end_time/time offset of the last hit object, in microsecond
    """
    last_obj = beatmap.hit_objects()[-1]
    if isinstance(last_obj, slider.beatmap.Slider):
        return last_obj.end_time / timedelta(microseconds=1)
    elif isinstance(last_obj, slider.beatmap.Spinner):
        return last_obj.end_time / timedelta(microseconds=1)
    else:
        # Circle
        # https://osu.ppy.sh/wiki/zh/Client/File_formats/Osu_%28file_format%29
        return last_obj.time / timedelta(microseconds=1)


def get_empty_beatmap():
    return slider.Beatmap.parse(empty_beatmap)


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

