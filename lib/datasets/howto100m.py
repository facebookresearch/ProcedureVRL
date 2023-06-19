# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
import numpy as np
import json
import lib.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
import pickle as pkl
logger = logging.get_logger(__name__)
import re
import pandas as pd
import ffmpeg
import math
import copy
import ipdb

def check_time(s1,e1,s2,e2):
    return max(min(e1,e2) - max(s1,s2),0)

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:  # training and validation
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:  # testing
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx

@DATASET_REGISTRY.register()
class Howto100m_develop(torch.utils.data.Dataset):
    """
    Dataloader to load video clips from HowTo100M dataset or COIN dataset.
    """
    def __init__(self, cfg, mode, num_retries=20):
        assert mode in ["train", "val", "test"], "Split '{}' not supported".format(mode)
        self.mode = mode
        self.cfg = copy.deepcopy(cfg)
        self._video_meta = {}
        self._num_retries = num_retries

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        # load multiple video segments with each one has NUM_FRAMES frames
        if hasattr(self.cfg.MODEL, 'NUM_SEG') and self.cfg.MODEL.NUM_SEG > 0:            
            self.cfg.DATA.NUM_FRAMES *= self.cfg.MODEL.NUM_SEG
        self.clip_feat_path = cfg.DEV.CLIP_VIS_FEAT_PATH  # pre-computed CLIP visual features

        # order pretraining related
        self.order_pretrain = cfg.DEV.ORDER_PRETRAIN_ENABLED
        self.order_max_len = cfg.DEV.ORDER_PRETRAIN_MAX_LEN
        self.order_stride = cfg.DEV.ORDER_STRIDE
        self.order_fix_recognition = cfg.DEV.ORDER_FIX_RECOGNITION
        self.order_recog_batch = cfg.DEV.ORDER_RECOG_BATCH
        
        # caption / text related
        if len(cfg.TRAIN.TEXT) > 0: # pretraining
            self.caps = cfg.TRAIN.TEXT

            from clip import tokenize # CLIP encoder
            self.tokenizer = tokenize
            if hasattr(cfg.MODEL, 'MAX_LEN'):
                self.max_len = cfg.MODEL.MAX_LEN
                print('self.max_len',self.max_len)
            else:        
                self.max_len = 64
            if hasattr(cfg.MODEL, 'MIN_LEN'):
                self.min_len = cfg.MODEL.MIN_LEN
                print('self.min_len',self.min_len)
            else:        
                self.min_len = 0   
        else: # finetuning
            self.caps = None
        
        if hasattr(cfg.TRAIN,  'EPOCH_MUL'):   
            self.em = cfg.TRAIN.EPOCH_MUL

        logger.info("Constructing dataset {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # label file
        path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode))
        assert PathManager.exists(path_to_file), "{} dir not found".format(path_to_file)

        # map video file root name into the actual name with extension in disk (to save initialization time)
        path_map_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "{}_{}.pth".format('path_map_dict', self.mode))
        if os.path.exists(path_map_file):
            path_map_exist = True
            path_map_dict = torch.load(path_map_file)
            logger.info("Load {} video file names with extension at {}".format(len(path_map_dict), str(path_map_file)))
        else:
            logger.info("Load video file names from scratch.")
            path_map_exist = False
            path_map_dict = {} 

        self._path_to_videos = []
        self._labels = []
        self._durations = []
        self._start = []
        self._end = []
        self._spatial_temporal_idx = []

        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                # show loading progress
                if clip_idx % 5000 == 0: 
                    print(clip_idx)

                # type of labels
                label_sep = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)
                len_label_sep = len(label_sep)
                if len_label_sep == 3: # task classification
                    path, label, duration = label_sep
                elif len_label_sep == 5: # step recognition, next step prediction
                    path, label, duration, start, end = label_sep
                else: 
                    path, label, duration, start, end, text = label_sep

                # construct dataset list
                for idx in range(self._num_clips):
                    # 1. get video name with extension
                    path = path.split('.')[0]
                    if path_map_exist: # pre-computed file name
                        if path in path_map_dict:
                            path = path_map_dict[path]
                        else: # no video exists, skip this video to save time
                            break
                    else: # find the actual extension of this video file in the folder
                        for extension in [".webm", ".mkv", ".mp4", ".m4a"]:
                            if os.path.exists(os.path.join(self.cfg.DATA.PATH_PREFIX, path+extension)):
                                path = path + extension
                                break
                        # no video exists, skip this video to save time
                        if '.' not in path:
                            break 
                        else:
                            path_map_dict[path.split('.')[0]] = path
                    
                    # 2. save metadata of this video
                    self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                    self._labels.append(int(label))
                    self._durations.append(int(float(duration)))
                    self._spatial_temporal_idx.append(idx)  # the local index among multiple views/crops of the same video clip
                    self._video_meta[clip_idx * self._num_clips + idx] = {}  # the global index among all counterparts of video clips

                    # 3. start, end, and text of this video clip
                    if len_label_sep == 3:
                        self._start.append(None)
                        self._end.append(None)
                    else:    
                        self._start.append(int(float(start)))
                        self._end.append(int(float(end)))

                # fast loading for debugging
                if self.cfg.DEV.LOAD_DUMMY_DATA and len(self._path_to_videos) > 50:
                    break

        assert (len(self._path_to_videos) > 0), "Failed to load split {} from {}".format(self._split_idx, path_to_file)
        logger.info("Constructing dataloader (size: {}) from {}".format(len(self._path_to_videos), path_to_file))
        
        # save the pre-computed file names to local disk
        if not path_map_exist:
            torch.save(path_map_dict, path_map_file)
            logger.info("Save {} video file names with extension at {}".format(len(path_map_dict), str(path_map_file)))
    
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        #########################################
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index
        if hasattr(self,'em') and self.em > 1:
            index = index % len(self._path_to_videos)
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S))
            if self.cfg.MULTIGRID.DEFAULT_S > 0: # decreasing the scale is equivalent to using a larger "span" in a sampling grid.
                min_scale = int(round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))
        elif self.mode in ["test"]:
            temporal_sample_index = self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left, center, or right if width is larger than height, and top, middle, or bottom if height is larger than width.
            spatial_sample_index = self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3 if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))
        sampling_rate = utils.get_random_sampling_rate(self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE, self.cfg.DATA.SAMPLING_RATE,)
        #########################################

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            # get video id
            vidid = self._path_to_videos[index].split('/')[-1].split('.')[0] # index might change over i_try
            
            # get the predefined start/end time of a video clip
            duration = self._durations[index]
            start = self._start[index]
            end = self._end[index]
            words = None
            text = None

            #########################################
            # when ASR is available: if the start/end time of a video clip is unknown (known), then randomly pick an ASR (get the corresponding ASR)
            if len(self.cfg.TRAIN.TEXT) > 0: # pretraining
                # ASR file
                cap = pd.read_csv(self.caps + vidid + '.csv')

                # determine which ASR and its index, since there is no start/end for video clips (HowTo100M)
                if self.order_pretrain: # if enable order pretraining, reserve the later ASR that are close to end of video
                    ind = random.randint(0, max(1, len(cap) - 1 - self.order_max_len * self.order_stride))
                else:
                    ind = random.randint(0, len(cap) - 1)
                
                # if enable order pretraining, get a sequence of video clips starting from the sampled/argmax clip above
                if self.order_pretrain: # multiple consective clips
                    # use ASR to construct a sequence of video clips
                    text, start, end = [], [], []
                    for seq_i in range(self.order_max_len):
                        this_ind = ind + seq_i * self.order_stride
                        if this_ind >= len(cap): # if the whole video is too short to cover enough clips, then duplicate the last one (for now)
                            this_ind = len(cap) - 1
                        # determine the start/end time of ASR
                        text_start_i, text_end_i, text_i = self.get_asr_start_end(cap, this_ind, vidid)
                        # refine the start/end time of this video segment based on ASR start/end time
                        start_i, end_i = self.get_video_start_end(text_start_i, text_end_i, duration, temporal_sample_index)
                        # add all into lists
                        text.append(text_i['clip_text_ids'].unsqueeze(0)); start.append(start_i); end.append(end_i)
                    text = {'clip_text_ids': torch.cat(text, dim=0)}
                else: # only one clip
                    # determine the start/end time of ASR
                    text_start, text_end, text = self.get_asr_start_end(cap, ind, vidid)
                    # refine the start/end time of this video segment based on ASR start/end time
                    start, end = self.get_video_start_end(text_start, text_end, duration, temporal_sample_index)
            else:
                # refine the start/end time of this video segment
                start, end = self.get_video_start_end(start, end, duration, temporal_sample_index)

            # step forecasting: if the end of video clip is fixed, such as step forecasting, load the exact start/end time from data csv file
            if hasattr(self.cfg.DATA,'FIX_END') and self.cfg.DATA.FIX_END:
                start = self._start[index]
                end = self._end[index]
                if self.cfg.DATA.FD < end - start: # video is too long
                    start = end - self.cfg.DATA.FD  
            #########################################
            
            #########################################
            # given timestamps of start and end, load video and sample raw video frames
            if self.order_pretrain: # multiple consecutive clips
                clips_frames = []
                for start_i, end_i in zip(start, end):
                    frames = self.get_frames(index, sampling_rate, temporal_sample_index, min_scale, duration, start_i, end_i, i_try, \
                                    spatial_sample_index, max_scale, crop_size)
                    if frames is None: 
                        break
                    else: 
                        clips_frames.append(frames.unsqueeze(0))
                frames = None if frames is None else torch.cat(clips_frames, dim=0)
            else: # single clip
                frames = self.get_frames(index, sampling_rate, temporal_sample_index, min_scale, duration, start, end, i_try, \
                                spatial_sample_index, max_scale, crop_size)
            # if decoding failed (wrong format, video is too short, and etc), select another video.
            if frames is None:  
                index = self.sample_new_index(index, i_try)
                continue
            
            # load label
            label = self._labels[index]
            #########################################
            
            # load ASR for creating pseudo labels
            if len(self.cfg.TRAIN.TEXT) > 0: # pretraining
                if text == None:
                    text = {'text': words}
                text['label'] = torch.tensor([1]+[0]*0)
                
                # get CLIP visual features given start and end time
                try:
                    if self.order_pretrain: # multiple consective clips
                        this_video = torch.load(self.clip_feat_path + vidid + '.pth') # load only once
                        frame_feats_lst = []
                        for start_i, end_i in zip(start, end):
                            frame_feats = get_clip_feat(self.clip_feat_path, vidid, start_i, end_i, this_video=this_video)
                            frame_feats_lst.append(torch.mean(frame_feats, dim=0).unsqueeze(0)) # mean pooling
                        text['clip_vis_feat'] = torch.cat(frame_feats_lst, dim=0)
                    else:
                        frame_feats = get_clip_feat(self.clip_feat_path, vidid, start, end)
                        text['clip_vis_feat'] = torch.mean(frame_feats, dim=0) # mean pooling
                except Exception as e:
                    logger.warning("{} -- CLIP feature: failed on video {} with start {} and end {}.".format(str(e), vidid, start, end))
                    dim = 512
                    if self.order_pretrain: # multiple consective clips
                        text['clip_vis_feat'] = torch.FloatTensor(self.order_max_len, dim).zero_() # dummy feature
                    else: # single clip
                        text['clip_vis_feat'] = torch.FloatTensor(dim).zero_() # dummy feature

                return frames, label, index, text
            
            return frames, label, index, {}

    def get_asr_start_end(self, cap, ind, vidid):
        """ Given captions and caption index, return the start/end time of ASR, tokenized ASR and its embedding (if any)
        """
        # if ASR is shorter than min_len, expand it to left (before) and right (later)
        if hasattr(self, 'min_len') and self.min_len > 0:    
            mi = 0
            q = cap['text'].values[ind]
            q = q if isinstance(q, str) else ' '
            s = cap['start'].values[ind]
            e = cap['end'].values[ind]
            while len(q.split(' ')) < self.min_len:
                if ind - mi > 0 and isinstance(cap['text'].values[ind-mi],str): # expand left
                    q = cap['text'].values[ind-mi]+ ' ' +q
                    s = cap['start'].values[ind-mi]
                if ind + mi < len(cap) and isinstance(cap['text'].values[ind+mi],str): # expand right
                    q = q + ' ' +cap['text'].values[ind+mi]   
                    e = cap['end'].values[ind+mi]
                mi += 1    
                if not ind - mi > 0 and not ind + mi < len(cap):
                    break  
            expanded_text, expanded_start, expanded_end = q, s, e # expanded text, start time, end time
        else:
            expanded_text, expanded_start, expanded_end = cap['text'].values[ind], cap['start'].values[ind], cap['end'].values[ind]
        
        # get ASR and tokenization
        sen = expanded_text
        if not type(sen) == type(' ') or len(sen) == 0:
            sen = ' '
        # tokenize ASR sentence using CLIP
        text = {'clip_text_ids': self.tokenizer([sen], truncate=True)}
            
        return expanded_start, expanded_end, text
    
    def get_video_start_end(self, start, end, duration, temporal_sample_index):
        """ Refine the start/end time of this video segment
        """
        if start == None:
            start, end = get_start_end_idx(duration, self.cfg.DATA.FD, temporal_sample_index, self.cfg.TEST.NUM_ENSEMBLE_VIEWS)
        # if the clip is too short, then expand it from the center to total duration as DATA.FD
        if end - start < self.cfg.DATA.FD - 1: 
            start = max((end+start)/2. - self.cfg.DATA.FD/2., 0)
            end = min(start+self.cfg.DATA.FD, duration)
        try:
            # if the clip is too long (than either NUM_FRAMES or DATA.FD) 
            if self.cfg.DATA.FD == 0. and end - start > self.cfg.DATA.NUM_FRAMES:
                new_end = (end+start)/2. + self.cfg.DATA.NUM_FRAMES/2.
                new_start = (end+start)/2. - self.cfg.DATA.NUM_FRAMES/2.
                start = new_start
                end = new_end
            elif self.cfg.DATA.FD > 0. and end - start > self.cfg.DATA.FD:
                startb, endb = start, end
                start, end = get_start_end_idx(end-start, self.cfg.DATA.FD, temporal_sample_index, self.cfg.TEST.NUM_ENSEMBLE_VIEWS)
                start += startb
                end += startb
        except:
            end = end
        return start, end

    def get_frames(self, index, sampling_rate, temporal_sample_index, min_scale, duration, start, end, i_try, spatial_sample_index, max_scale, crop_size):
        """ Load video clip given start and end time
        """
        # ffmpeg loading
        try:
            frames = get_video(self._path_to_videos[index], start, end, self.cfg.DATA.NUM_FRAMES)
        except:
            frames = None

        # If decoding failed (wrong format, video is too short, and etc), select another video.
        if frames is None:
            return None

        # Perform color normalization.
        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)

        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        if self.cfg.MODEL.ARCH not in ['vit', 'swin3d', 'mvit']:
            frames = utils.pack_pathway_output(self.cfg, frames)
        return frames
 
    def sample_new_index(self, index, i_try):
        logger.warning("Failed to decode video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))
        if self.mode not in ["test"]:
            index = random.randint(0, len(self._path_to_videos) - 1)
        if self.mode in ["test"] and i_try > self._num_retries // 2:
            index = random.randint(0, len(self._path_to_videos) - 1)
        return index

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        if hasattr(self, 'em') and self.em > 1 and self.mode == 'train':
            return len(self._path_to_videos) * self.em
        else: 
            return len(self._path_to_videos)
        
def get_clip_feat(clip_feat_path, vidid, start, end, this_video=None):
    """ Given start and end time, fetch clip features from local disk and return frame-level features
    """
    if this_video is None:
        this_video = torch.load(clip_feat_path + vidid + '.pth')
    mid_time = this_video['mid_time']
    # determine indices of start and end frame
    start = math.ceil(start)
    end = math.floor(end)
    if start in mid_time:
        start_ind = mid_time.index(start)
    else:
        start_ind = mid_time.index(start + 1)
    if end in mid_time:
        end_ind = mid_time.index(end)
    else:
        end_ind = mid_time.index(end - 1)

    # get clip-level feature
    frame_feats = this_video['clip_instances'][start_ind:(end_ind+1)]   
    frame_feats = [torch.unsqueeze(x, 0) for x in frame_feats] # add dimension
    frame_feats = torch.cat(frame_feats, dim=0)
    frame_feats = frame_feats.float() # fp16 is the dtype in disk
    return frame_feats
    
def get_video(video_path, start, end, number_frames):
    cmd = (
        ffmpeg
        .input(video_path, ss=start, t=end-start)
        .filter('fps', fps=math.ceil(number_frames/(end-start)))
    )
    cmd = (
            cmd.filter('scale', 640, 360)
        )
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=False, quiet=True)
    )
    
    video = np.frombuffer(out, np.uint8).reshape([-1, 360, 640, 3])
    video2 = torch.tensor(video)
    video2 = temporal_sampling(video2, 0, video2.shape[0], number_frames)
    return video2  

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames