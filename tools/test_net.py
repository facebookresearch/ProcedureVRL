# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
import cv2
from einops import rearrange, reduce, repeat
import scipy.io

import lib.utils.checkpoint as cu
import lib.utils.distributed as du
import lib.utils.logging as logging
import lib.utils.misc as misc
import lib.visualization.tensorboard_vis as tb
from lib.datasets import loader
from lib.models import build_model
from lib.utils.meters import TestMeter, EPICTestMeter
import torch.nn.functional as F
import ipdb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.get_logger(__name__)
allgather = du.AllGather.apply

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
    all_distance = []
    if cfg.TEST.DATASET in ['Epickitchens']:
        results_lst = []

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS: # use GPUs
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], (str,)):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()
        
        if isinstance(labels, (dict,)): # EPIC-Kitchens dataset
            preds = model(inputs)
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                verb_preds, verb_labels, video_idx = du.all_gather([preds[0], labels['verb'], video_idx])
                noun_preds, noun_labels, video_idx = du.all_gather([preds[1], labels['noun'], video_idx])
                meta = du.all_gather_unaligned(meta)
                metadata = {'narration_id': []}
                for i in range(len(meta)):
                    metadata['narration_id'].extend(meta[i]['narration_id'])
                
                # collect results
                results_lst.append({'verb_output': verb_preds.cpu(), 'noun_output': noun_preds.cpu(), 
                                    'verb_label': verb_labels.cpu(), 'noun_label': noun_labels.cpu(),
                                    'narration_id': metadata['narration_id']})
            else:
                metadata = meta
                verb_preds, verb_labels, video_idx = preds[0], labels['verb'], video_idx
                noun_preds, noun_labels, video_idx = preds[1], labels['noun'], video_idx
            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats((verb_preds.detach().cpu(), noun_preds.detach().cpu()),
                (verb_labels.detach().cpu(), noun_labels.detach().cpu()), metadata, video_idx.detach().cpu(),)
            test_meter.log_iter_stats(cur_iter)    
        else:
            # Perform the forward pass.
            preds = model(inputs)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1 and not (cfg.TRAIN.LABEL_EMB == '' and cfg.TRAIN.TEXT != ''):
                preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds.detach(), labels.detach(), video_idx.detach(), video_name_list=test_loader.dataset._path_to_videos)
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()
    
    # Log epoch stats and print the final testing results.
    if cfg.TEST.DATASET in ['Epickitchens']:
        if du.is_master_proc():
            # results = {'verb_output': preds[0],
            #         'noun_output': preds[1],
            #         'narration_id': metadata}
            scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            TEST_SPLIT = "validation"
            file_path = os.path.join(scores_path, TEST_SPLIT + '.pkl')
            pickle.dump(results_lst, open(file_path, 'wb'))
    else:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        all_video_names = test_meter.video_names
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()

        torch.save({'names': all_video_names, 'preds': all_preds, 'labels': all_labels}, 'vis_pred_zeroshot_step_cls.pth') # , 'vis_pred_zeroshot.pth') #  , 'vis_pred_finetuned.pth') # , 'vis_pred.pth') # 
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)
        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            with PathManager.open(save_path, "wb") as f:
                pickle.dump([all_labels, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))
    
    test_meter.finalize_metrics(ks=(1, 5)) # top-1 and top-5 accuracy / recall
    return test_meter


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

    assert len(test_loader.dataset) % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0
    # Create meters for multi-view testing.
    if cfg.TEST.DATASET in ['Epickitchens']:
        test_meter = EPICTestMeter(
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            [97, 300],
            len(test_loader),
        )
    else:
        test_meter = TestMeter(
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES if cfg.TRAIN.LABEL_EMB == '' or cfg.TRAIN.TEXT == '' else 1059,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
