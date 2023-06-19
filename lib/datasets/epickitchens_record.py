#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from .video_record import VideoRecord
from datetime import timedelta
import time
import ipdb

def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 100
    return sec


class EpicKitchensVideoRecord(VideoRecord):
    def __init__(self, tup, enable_anticipation=False, fd=0.0):
        self._index = str(tup[0])
        self._series = tup[1]
        self.enable_anticipation = enable_anticipation
        self.fd = fd # fixed duration for anticipation

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        if self.enable_anticipation:
            start_t = max(0.0, (self.end_frame / float(self.fps)) - self.fd) # take a self.fd seconds of video clip for anticipation
            return int(round(start_t * self.fps))
        else:
            return int(round(timestamp_to_sec(self._series['start_timestamp']) * self.fps))

    @property
    def end_frame(self):
        if self.enable_anticipation:
            end_t = timestamp_to_sec(self._series['start_timestamp']) - 1.0 # 1 second of anticipation time
            return int(round(end_t * self.fps))
        else:
            return int(round(timestamp_to_sec(self._series['stop_timestamp']) * self.fps))

    @property
    def fps(self):
        is_100 = len(self.untrimmed_video_name.split('_')[1]) == 3
        return 50 if is_100 else 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def label(self):
        return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -1}

    @property
    def metadata(self):
        return {'narration_id': self._index}