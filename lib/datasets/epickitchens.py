#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import numpy as np
import math
import ffmpeg
from pytorchvideo.data.encoded_video import EncodedVideo
import random
import os
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms

import lib.utils.logging as logging
from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensVideoRecord
from . import autoaugment as autoaugment
from . import transform as transform
from . import utils as utils
# from .frame_loader import pack_frames_to_video_clip
from .decoder import get_start_end_idx

import copy
import ipdb

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Epickitchens(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        assert mode in ["train", "val", "test", "train+val"], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = copy.deepcopy(cfg)
        self.mode = mode
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.target_fps = cfg.DATA.TARGET_FPS
        self.fd = self.cfg.DATA.FD
        self._num_retries = 10
        self.use_bgr_order = True

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)

        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TRAIN_LIST)]
        elif self.mode == "val":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
        elif self.mode == "test":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        else:
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file) for file in [self.cfg.EPICKITCHENS.TRAIN_LIST, self.cfg.EPICKITCHENS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(file)
        
        self._video_records = []
        self._spatial_temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    this_record = EpicKitchensVideoRecord(tup) # video_id, start/stop_timestamp, start/stop_frame, narration, verb, noun
                    self._video_records.append(this_record) 
                    self._spatial_temporal_idx.append(idx)

        assert len(self._video_records) > 0, "Failed to load EPIC-KITCHENS split {} from {}".format(self.mode, path_annotations_pickle)
        logger.info("Constructing epickitchens dataloader (size: {}) from {}".format(len(self._video_records), path_annotations_pickle))

    def __len__(self):
        return len(self._video_records)
    
    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._video_records)

    def sample_new_index(self, index, i_try):
        logger.warning("Failed to decode video idx {} from {}; trial {}".format(index, self._video_records[index].untrimmed_video_name, i_try))
        if self.mode not in ["test"]:
            index = random.randint(0, len(self._video_records) - 1)
        if self.mode in ["test"] and i_try > self._num_retries // 2:
            index = random.randint(0, len(self._video_records) - 1)
        return index

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
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(10):
            if self.mode in ["train", "val", "train+val"]:
                # -1 indicates random sampling.
                temporal_sample_index = -1
                spatial_sample_index = -1
                min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
                max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            elif self.mode in ["test"]:
                temporal_sample_index = self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
                # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
                # center, or right if width is larger than height, and top, middle,
                # or bottom if height is larger than width.
                if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                    spatial_sample_index = self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                    spatial_sample_index = 1
                min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
                # The testing is deterministic and no jitter should be performed.
                # min_scale, max_scale, and crop_size are expect to be the same.
                assert len({min_scale, max_scale, crop_size}) == 1
            else:
                raise NotImplementedError("Does not support {} mode".format(self.mode))
            
            try:
                frames = pack_frames_to_video_clip(self.cfg, self.num_frames, self._video_records[index], temporal_sample_index,\
                        target_fps=self.target_fps, mode=self.mode, use_bgr_order=self.use_bgr_order,)
            except:
                index = self.sample_new_index(index, i_try)
                continue

            if self.cfg.DATA.USE_RAND_AUGMENT and self.mode in ["train"]:
                # Transform to PIL Image
                frames = [transforms.ToPILImage()(frame.squeeze().numpy()) for frame in frames]

                # Perform RandAugment
                img_size_min = crop_size
                auto_augment_desc = "rand-m15-mstd0.5-inc1"
                aa_params = dict(translate_const=int(img_size_min * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in self.cfg.DATA.MEAN]))
                seed = random.randint(0, 100000000)
                frames = [autoaugment.rand_augment_transform(auto_augment_desc, aa_params, seed)(frame) for frame in frames]

                # To Tensor: T H W C
                frames = [torch.tensor(np.array(frame)) for frame in frames]
                frames = torch.stack(frames)
            
            # Perform color normalization.
            #frames = utils.tensor_normalize(
            frames = tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)

            # Perform data augmentation.
            use_random_resize_crop = self.cfg.DATA.USE_RANDOM_RESIZE_CROPS
            if use_random_resize_crop:
                if self.mode in ["train", "val"]:
                    frames = transform.random_resize_crop_video(frames, crop_size, interpolation_mode="bilinear")
                    frames, _ = transform.horizontal_flip(0.5, frames)
                else:
                    assert len({min_scale, max_scale, crop_size}) == 1
                    frames, _ = transform.random_short_side_scale_jitter(frames, min_scale, max_scale)
                    frames, _ = transform.uniform_crop(frames, crop_size, spatial_sample_index)
            else:
                # Perform data augmentation.
                #frames = utils.spatial_sampling(
                frames = spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
            
            # T H W C -> T C H W.
            if self.mode in ["train", "val"]:
                frames = frames.permute(1, 0, 2, 3) # C T H W -> T C H W
                #frames = utils.frames_augmentation(
                frames = frames_augmentation(
                    frames,
                    colorjitter=self.cfg.DATA.COLORJITTER,
                    use_grayscale=self.cfg.DATA.GRAYSCALE,
                    use_gaussian=self.cfg.DATA.GAUSSIAN
                )

            label = self._video_records[index].label
            #frames = utils.pack_pathway_output(self.cfg, frames)
            frames = pack_pathway_output(self.cfg, frames)
            metadata = self._video_records[index].metadata
            return frames[0], label, index, metadata

def pack_frames_to_video_clip(cfg, num_frames, video_record, temporal_sample_index, target_fps=30, mode=None, use_bgr_order=False):
    # metadata
    fps = video_record.fps
    sampling_rate = cfg.DATA.SAMPLING_RATE
    num_samples = num_frames

    # temporally sample views (with desired duration) from original video clip (could be longer/shorter)
    start_idx, end_idx = get_start_end_idx(
        video_record.num_frames,
        num_samples * sampling_rate * fps / target_fps, # 32 * 2 * 60 / 30 (2 seconds, 16 frames per second)
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
    )
    start_idx, end_idx = start_idx + 1, end_idx + 1
    # uniformly sample num_samples frames
    frame_idx = temporal_sampling(
        video_record.num_frames,
        start_idx, end_idx, num_samples,
        start_frame=video_record.start_frame,
    )

    ############################################
    # NOTE: EncodedVideo and ffmpeg returns RGB format, but cv2.imdecode() returns BGR format from extracted rgb frames
    ############################################
    # convert the start/end frame id (after view sampling) back to timestamps
    time_stamps = frame_idx / float(video_record.fps)
    relative_time_stamps = (frame_idx - video_record.start_frame) / float(video_record.fps)
    
    # full video
    video_name = os.path.join(cfg.EPICKITCHENS.VISUAL_DATA_DIR, video_record.untrimmed_video_name) + '.mp4'
    start_sec = time_stamps[0].item()
    end_sec = time_stamps[-1].item()
    video = EncodedVideo.from_path(file_path=video_name, decode_audio=False)
    
    # extract frames from video
    vid_out_cthw = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
    vid_out_thwc = vid_out_cthw.permute(1, 2, 3, 0)
    
    # sample frames if #frames > number_frames
    index = torch.linspace(0, vid_out_thwc.shape[0], num_frames)
    index = torch.clamp(index, 0, vid_out_thwc.shape[0] - 1).long()
    frames = torch.index_select(vid_out_thwc, 0, index)
    frames = frames.type(torch.uint8)
    if use_bgr_order:
        frames = frames[:, :, :, [2,1,0]]

    return frames

def temporal_sampling(num_frames, start_idx, end_idx, num_samples, start_frame=0):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, num_frames - 1).long()
    return start_frame + index

def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        if cfg.DISTILLATION.TEACHER_MODEL == "R2Plus1D_34":
            fast_pathway = torch.nn.functional.interpolate(
                fast_pathway[:, :32, :, :].permute(1, 0, 2, 3), size=(112, 112), mode='bilinear', align_corners=True)
            fast_pathway = fast_pathway.permute(1, 0, 2, 3)
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list

def spatial_sampling(frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224, random_horizontal_flip=True, inverse_uniform_sampling=False):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(images=frames, min_size=min_scale, max_size=max_scale, inverse_uniform_sampling=inverse_uniform_sampling)
        frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames

def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def frames_augmentation(frames, colorjitter=True, use_grayscale=True, use_gaussian=False):
    if colorjitter:
        if np.random.uniform() >= 0.2:
            frames = transform.color_jitter(frames, 0.4, 0.4, 0.4)

    # Perform gray-scale with prob=0.2
    if use_grayscale:
        if np.random.uniform() >= 0.8:
            frames = transform.grayscale(frames)

    # Perform gaussian blur with prob=0.5
    if use_gaussian:
        if np.random.uniform() >= 0.5:
            frames = transform.gaussian_blur(frames)

    # T C H W -> C T H W
    frames = frames.permute(1, 0, 2, 3)

    return frames    