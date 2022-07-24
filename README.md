Generates osu! beatmaps.
## Usage:
1. find `./resources/config/inference/cganv4.yaml`, change the line `weights_path: xx` to the path to model weight file
2. goto `./run_generate.py`,
   * change `audio_file_path` to the path to audio file,
   * set `audio_info_path` to `.yaml` which records audio information(bpm, start beat time...), just set it to `None` on first run to allow automatic information extraction,
   * edit meta info in `meta_list`,
   * run it and `.osz` will be generated under `./resources/gen/osz`.
   
3. after first run, you may set `audio_info_path` to corresponding `.yaml` under `./resources/gen/audio_info` to accelerate generation for this audio.
   
### What you should know about osu! beatmaps before generating
one `.osz` for one beatmapset, one `.osu` for one beatmap(may be of different difficulties),
all beatmaps in one beatmapset share one audio file.
