# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import lib.models.losses as losses
import lib.models.optimizer as optim
import lib.utils.checkpoint as cu
import lib.utils.distributed as du
import lib.utils.logging as logging
import lib.utils.metrics as metrics
import lib.utils.misc as misc
import lib.visualization.tensorboard_vis as tb
from lib.models.losses import ActLocMSELoss
from lib.datasets import loader
from lib.models import build_model
from lib.utils.meters import TrainMeter, ValMeter, EPICTrainMeter, EPICValMeter
from lib.utils.multigrid import MultigridSchedule

#from timm.data import Mixup
from lib.datasets import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from einops import rearrange, reduce, repeat

import torch.nn.functional as F
import ipdb

logger = logging.get_logger(__name__)

allgather = du.AllGather.apply

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))
    

def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    if cfg.TRAIN.LINEAR:
        model.train()
        if cfg.NUM_GPUS > 1:
            if hasattr(model.module.model, "pos_drop"):
                model.module.model.pos_drop.eval()
                model.module.model.blocks.eval()
            elif hasattr(model.module.model, "video_encoder"): # mvit-v2
                model.module.model.video_encoder.eval()
        else:
            if hasattr(model.model, "pos_drop"):
                model.model.pos_drop.eval()
                model.model.blocks.eval()  
            elif hasattr(model.model, "video_encoder"): # mvit-v2
                model.model.video_encoder.eval()
    else:    
        model.train()
    
    # NOTE: no gradients applied on text model
    if cfg.NUM_GPUS > 1:
        if hasattr(model.module.model, 'text_model'):
            model.module.model.text_model.eval()
    else:
        if hasattr(model.model, 'text_model'):
            model.model.text_model.eval()
    train_meter.iter_tic()
    data_size = len(train_loader)

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size
    for cur_iter, (inputs, labels, indexes, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            if labels.dim() == 2:
                labels = labels.view(-1)
            labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    if not isinstance(val[i], (str,)):
                        val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        # Explicitly declare reduction to mean.
        if cfg.MODEL.LOSS_FUNC == 'smooth':
            loss_fun = LabelSmoothingCrossEntropy(0.2)
        elif cfg.MODEL.LOSS_FUNC == 'kldiv':
            loss_fun_1 = torch.nn.KLDivLoss(reduction='batchmean')  
            loss_fun_2 = torch.nn.MSELoss(reduction='mean')
        elif not cfg.MIXUP.ENABLED:
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            loss_fun_2 = torch.nn.MSELoss(reduction='mean')
        else:
            mixup_fn = Mixup(mixup_alpha=cfg.MIXUP.ALPHA, cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA, cutmix_minmax=cfg.MIXUP.CUTMIX_MINMAX, 
                prob=cfg.MIXUP.PROB, switch_prob=cfg.MIXUP.SWITCH_PROB, mode=cfg.MIXUP.MODE, label_smoothing=0.1, num_classes=cfg.MODEL.NUM_CLASSES)
            hard_labels = labels   
            inputs, labels = mixup_fn(inputs, labels)
            loss_fun = SoftTargetCrossEntropy()

        # Model forward
        if cfg.TRAIN.LABEL_EMB != '' and cfg.TRAIN.TEXT != '': # pretraining
            meta = {k:meta[k].view(-1, meta[k].shape[-1]) for k in meta}
            pred, teacher_pred, mse_loss = model([inputs, meta]) # pred: logits over all steps
        else: # finetuning
            preds = model(inputs) # already applied softmax

        # Calculate loss
        if cfg.TRAIN.LABEL_EMB != '' and cfg.TRAIN.TEXT != '': # pretraining
            with torch.no_grad():
                preds = F.softmax(pred, 1)
                teacher_pred = F.softmax(teacher_pred,1)
                if cfg.TRAIN.TOPK != 0: # get probabilities of top-k entries
                    teacher_pred = (teacher_pred.unsqueeze(1)*(teacher_pred.unsqueeze(1)==teacher_pred.topk(k=cfg.TRAIN.TOPK,dim=1)[0].unsqueeze(2)).float()).sum(1)
                    teacher_pred = teacher_pred / teacher_pred.sum(1, keepdim=True)
            if cfg.MODEL.LOSS_FUNC == 'kldiv':  # KL loss with top-k entries
                loss1 = loss_fun_1(F.log_softmax(pred, dim=1), teacher_pred)  
                loss2 = loss_fun_2(mse_loss[0], mse_loss[1])
                loss = loss1 + loss2  
        elif isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "Epickitchens": # EPIK-Kitchens finetuning
            loss_verb = loss_fun(preds[0], labels['verb'])
            loss_noun = loss_fun(preds[1], labels['noun'])
            loss = 0.5 * (loss_verb + loss_noun)  
        else:  # finetuning
            loss = loss_fun(preds, labels)

        if cfg.MIXUP.ENABLED:
            labels = hard_labels

        # check Nan Loss.
        misc.check_nan_losses(loss)

        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE: # apply gradients
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()
        else: # accumulate gradients after #batch reaches cfg.GLOBAL_BATCH_SIZE
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward()
            if (cur_iter + 1) % num_iters == 0:
                for p in model.parameters():
                    if not p.grad == None:
                        p.grad /= num_iters
                # Update the parameters
                optimizer.step()
                optimizer.zero_grad()


        if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "Epickitchens":
            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce([loss_verb, verb_top1_acc, verb_top5_acc])
            # Copy the stats from GPU to CPU (sync point).
            loss_verb, verb_top1_acc, verb_top5_acc = (loss_verb.item(), verb_top1_acc.item(), verb_top5_acc.item(),)

            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce([loss_noun, noun_top1_acc, noun_top5_acc])
            # Copy the stats from GPU to CPU (sync point).
            loss_noun, noun_top1_acc, noun_top5_acc = (loss_noun.item(), noun_top1_acc.item(), noun_top5_acc.item(),)

            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]), (labels['verb'], labels['noun']), (1, 5))
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, action_top1_acc, action_top5_acc = du.all_reduce([loss, action_top1_acc, action_top5_acc])
            # Copy the stats from GPU to CPU (sync point).
            loss, action_top1_acc, action_top5_acc = (loss.item(), action_top1_acc.item(), action_top5_acc.item(),)

            # Update and log stats.
            train_meter.update_stats((verb_top1_acc, noun_top1_acc, action_top1_acc), (verb_top5_acc, noun_top5_acc, action_top5_acc),
                (loss_verb, loss_noun, loss), lr, inputs[0].size(0) * cfg.NUM_GPUS)
        else:
            top1_err, top5_err = None, None
            # since labels during pretraining are not used, copy label to the same shape of prediction, for consistency
            if cfg.DEV.ORDER_PRETRAIN_ENABLED:
                labels = labels[0].expand(preds.size(0))

            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, min(5,preds.shape[0])))
            top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])
            # Copy the stats from GPU to CPU (sync point).
            loss, top1_err, top5_err = (loss.item(), top1_err.item(), top5_err.item(),)

            # Update and log stats.
            train_meter.update_stats(top1_err, top5_err, loss, lr, inputs[0].size(0) * max(cfg.NUM_GPUS, 1),)


        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    logger.info('\nStart eval_epoch!')

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    texts=[]
    vids=[]

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        #labels = labels.cuda()
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    if not isinstance(val[i], (str,)):
                        val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()
        
        # Model forward
        preds = model(inputs)

        if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "Epickitchens":
            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))
            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                verb_top1_acc, verb_top5_acc = du.all_reduce([verb_top1_acc, verb_top5_acc])
            # Copy the errors from GPU to CPU (sync point).
            verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))
            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                noun_top1_acc, noun_top5_acc = du.all_reduce([noun_top1_acc, noun_top5_acc])
            # Copy the errors from GPU to CPU (sync point).
            noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]), (labels['verb'], labels['noun']), (1, 5))
            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])
            # Copy the errors from GPU to CPU (sync point).
            action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()

            # Update and log stats.
            val_meter.iter_toc()
            val_meter.update_stats((verb_top1_acc, noun_top1_acc, action_top1_acc), (verb_top5_acc, noun_top5_acc, action_top5_acc),
                inputs[0].size(0) * cfg.NUM_GPUS)
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            # Combine the errors across the GPUs.
            top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            # Update and log stats.
            val_meter.iter_toc()
            val_meter.update_stats(top1_err, top5_err, inputs[0].size(0) * max(cfg.NUM_GPUS, 1),)


        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    if cfg.TRAIN.LABEL_EMB == '' and cfg.TRAIN.TEXT != '' and 'coin' in cfg.DATA.PATH_TO_DATA_DIR:
        all_preds = torch.mm(torch.cat(vids), torch.cat(texts).transpose(0,1))
        dis = all_preds.numpy().transpose()
        print(dis.shape)
        met = compute_metrics(dis)
        print_computed_metrics(met)

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """
    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True)

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (model, optimizer, train_loader, val_loader, precise_bn_loader, train_meter, val_meter,)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
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

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if not cfg.TRAIN.FINETUNE:
        start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
        start_epoch = 0
        cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)
    
    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True) if cfg.BN.USE_PRECISE_STATS else None
    
    # Create evaluator
    if cfg.TRAIN.DATASET == 'Epickitchens':
        train_meter = EPICTrainMeter(len(train_loader), cfg)
        val_meter = EPICValMeter(len(val_loader), cfg)
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    #eval_epoch(val_loader, model, val_meter, 0, cfg, writer)
    for name, param in model.named_parameters():
        if True: # not param.requires_grad:
            logger.info('{} {}'.format(name, str(param.requires_grad)))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                model, optimizer, train_loader, val_loader, precise_bn_loader, train_meter, val_meter = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer)

        is_checkp_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch, None if multigrid is None else multigrid.schedule,)
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch, None if multigrid is None else multigrid.schedule)

        # Compute precise BN stats.
        if ((is_checkp_epoch or is_eval_epoch) and cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0):
            calculate_and_update_precise_bn(precise_bn_loader, model, min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)), cfg.NUM_GPUS > 0,)
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
