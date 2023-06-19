# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import torch
import os
import pickle

from lib.utils.misc import launch_job
from lib.utils.parser import load_config, parse_args

import lib.utils.checkpoint as cu
import lib.utils.distributed as du
import lib.utils.logging as logging
import lib.utils.misc as misc
import lib.visualization.tensorboard_vis as tb
from lib.datasets import loader
from lib.models import build_model
from lib.utils.meters import TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    num_videos = len(test_loader.dataset) // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
#    video_preds = torch.zeros((num_videos, cfg.TEST.NUM_SPATIAL_CROPS, 2048))
#    video_clip_counter = torch.zeros((num_videos, 1))
#    video_preds = np.zeros((num_videos, cfg.DATA.NUM_FRAMES, cfg.TEST.NUM_SPATIAL_CROPS, 400))
#    video_preds = np.zeros((num_videos, cfg.TEST.NUM_SPATIAL_CROPS, cfg.MODEL.NUM_CLASSES))
    video_preds = np.zeros((num_videos, cfg.TEST.NUM_SPATIAL_CROPS * cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.MODEL.NUM_CLASSES))
    video_clip_counter = np.zeros(num_videos, dtype=np.int)

    video_labels = torch.zeros((num_videos, cfg.MODEL.NUM_CLASSES))
    num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
    print(f"n_video: {num_videos}; n_clip: {num_clips}")

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        labels = labels.cuda()
        video_idx = video_idx.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Perform the forward pass.
        preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )


        if du.is_master_proc():
            preds = preds.detach().cpu()
            labels = labels.detach().cpu()
            clip_ids = video_idx.detach().cpu()
            for ind in range(preds.shape[0]):
                vid_id = int(clip_ids[ind]) // num_clips
                #clip_id = int(video_clip_counter[vid_id])
                clip_id = int(clip_ids[ind]) % num_clips
                video_labels[vid_id] = labels[ind]
                video_preds[vid_id, clip_id, :] = preds[ind].numpy()
                video_clip_counter[vid_id] += 1

#        if du.is_master_proc():
#            preds = preds.detach().cpu()
#            labels = labels.detach().cpu()
#            clip_ids = video_idx.detach().cpu()
#            for ind in range(preds.shape[0]):
#                vid_id = int(clip_ids[ind]) // num_clips
#                video_labels[vid_id] = labels[ind]
#                video_preds[vid_id] += preds[ind]
#
        if cur_iter % 100 == 0:
            print(f"{cur_iter}/{len(test_loader)}", flush=True) 
        # break
#
#    video_preds /= num_clips

    #######
    if cfg.TEST.SAVE_PREDICT_PATH != "" and du.is_master_proc():
       #output_path = '/checkpoint/gedas/seq2seq_results/extracted_visual_feats/vit_8x8_base_patch16_224_sep1x2_lr5e-3_sgd_step_custom_wd1e-4_Hacs_ep25/X_test.npy'
       output_path = cfg.TEST.SAVE_PREDICT_PATH
       print(output_path)
       print(video_preds.shape)
       #print(np.mean(video_preds.flatten()))
       np.save(output_path, video_preds)
    ###########

#    if cfg.TEST.SAVE_PREDICT_PATH != "" and du.is_master_proc():
#        with open(
#            os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_PREDICT_PATH), "wb"
#        ) as f:
#            pickle.dump(
#                (
#                    video_labels.numpy(),
#                    video_preds.numpy(),
#                ),
#                f,
#            )
#            logger.info("saving predictions!")


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()


from lib.utils.parser import load_config, parse_args


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
