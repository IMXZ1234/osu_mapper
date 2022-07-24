Generates osu! beatmaps.

Currently, only timings of hit objects are predicted, positions are still generated at random under simple preset rules.

Predicting positions of hit objects may be much more difficult than predicting timings, which is remained for future work.

__This project is still under experimental status, You should first create a conda environment according to `./requirements.yaml` to proceed.__

### Usage (Generation):
1. find `./resources/config/inference/cganv4.yaml`, change the line `weights_path: xx` to the path to model weight file.
   you may use pretrained models under `./resources/pretrained_models`.
2. goto `./run_generate.py`.
   * change `audio_file_path` to the path to audio file.
   * set `audio_info_path` to the `.yaml` file which records audio information(bpm, start beat time...).
     on first run just set it to `None` to allow automatic information extraction.
   * edit meta info in `meta_list`.
   * run it and `.osz` will be generated under `./resources/gen/osz`.
   
3. after first run, you may set `audio_info_path` to corresponding `.yaml` under `./resources/gen/audio_info` to accelerate generation for this audio.
   
### What you should know about osu! beatmaps
* one `.osz` for one beatmapset, one `.osu` for one beatmap
* one beatmapset often contains multiple beatmaps, beatmaps in same beatmapset are generally of different difficulties.
* all beatmaps in one beatmapset share one audio file.

### Acknowledgements:
[Beatnet](https://github.com/mjhydri/BeatNet) is used to extract audio bpm and start beat, thanks for their great jobs.
