if __name__ == '__main__':
    generator = BeatmapGenerator(
        r'../resources/config/seg_mlp_bi_lr0.1.yaml',
        r'../resources/result/cnnv1/0.1/model_epoch5.pt'
    )
    # audio_file_path = r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\bgm\audio.mp3'
    audio_file_path = r'C:\Users\asus\AppData\Local\osu!\Songs\869801 Meramipop - Sacrifice\audio.mp3'
    out_path = r'..\resources\data\bgm\out.osu'
    audio_info_path = r'..\resources\data\bgm\audio_info.txt'
    generator.generate_beatmap(audio_file_path, 3, out_path, audio_info_path)
