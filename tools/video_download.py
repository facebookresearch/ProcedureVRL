# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

# COIN dataset
import json
import os

output_path = '/fsx/yiwuzhong/data/coin/videos/videos_flat'
json_path = '/fsx/yiwuzhong/data/coin/COIN.json'

if not os.path.exists(output_path):
	os.mkdir(output_path)
	
data = json.load(open(json_path, 'r'))['database']
print("We got {} videos in json files!".format(len(data)))
youtube_ids = list(data.keys())

for y_i, youtube_id in enumerate(data):  
    info = data[youtube_id]
    url = info['video_url']
    # type = info['recipe_type']
    vid_loc = output_path # + '/' + str(type) # no sub-folders
    if not os.path.exists(vid_loc):
        os.mkdir(vid_loc)
    try:
        # default extension '.webm'; write (automatically generated) subtitles as srt format and english only
        os.system('yt-dlp -o ' + vid_loc + '/' + youtube_id + ' --write-auto-subs --convert-subs srt --sub-langs en --geo-verification-proxy ' + url + ' --geo-bypass ' + url)
    except Exception as e:
        print(e)