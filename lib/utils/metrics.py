# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch
import numpy as np
from typing import Union, Dict, List, Tuple
import ipdb

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def multitask_topks_correct(preds, labels, ks=(1,)):
    """
    Args:
        preds: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        ks: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    ####
    # NOTE: function for calculating action accuracy
    ####
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
#     if torch.cuda.is_available():
#         all_correct = all_correct.cuda()
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        if not all_correct.device == correct_for_task.device:
            all_correct = all_correct.to(correct_for_task.device)
        all_correct.add_(correct_for_task) # ByteTensor addition (True + True = 2)
    
    # a correct predict means the prediction for each single task (verb, noun) is correct
    multitask_topks_correct = [
        torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks
    ]

    return multitask_topks_correct


def multitask_topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
   """
    num_multitask_topks_correct = multitask_topks_correct(preds, labels, ks)
    return [(x / preds[0].size(0)) * 100.0 for x in num_multitask_topks_correct]



#######################################################
######## EPIC-Kitchens-100 action anticipation ########
#######################################################
def softmax(xs):
    if xs.ndim == 1:
        xs = xs.reshape((1, -1))
    max_x = np.max(xs, axis=1).reshape((-1, 1))
    exp_x = np.exp(xs - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

def top_scores(scores: np.ndarray, top_n: int = 100):
    """
    Examples:
        >>> top_scores(np.array([0.2, 0.6, 0.1, 0.04, 0.06]), top_n=3)
        (array([1, 0, 2]), array([0.6, 0.2, 0.1]))
    """
    if scores.ndim == 1:
        top_n_idx = scores.argsort()[::-1][:top_n]
        return top_n_idx, scores[top_n_idx]
    else:
        top_n_scores_idx = np.argsort(scores)[:, ::-1][:, :top_n]
        top_n_scores = scores[
            np.arange(0, len(scores)).reshape(-1, 1), top_n_scores_idx
        ]
        return top_n_scores_idx, top_n_scores

def compute_action_scores(verb_scores, noun_scores, top_n=100):
    top_verbs, top_verb_scores = top_scores(verb_scores, top_n=top_n)
    top_nouns, top_noun_scores = top_scores(noun_scores, top_n=top_n)
    top_verb_probs = softmax(top_verb_scores)
    top_noun_probs = softmax(top_noun_scores)
    action_probs_matrix = (
        top_verb_probs[:, :, np.newaxis] * top_noun_probs[:, np.newaxis, :]
    )
    instance_count = action_probs_matrix.shape[0]
    action_ranks = action_probs_matrix.reshape(instance_count, -1).argsort(axis=-1)[
        :, ::-1
    ]
    verb_ranks_idx, noun_ranks_idx = np.unravel_index(
        action_ranks[:, :top_n], shape=(action_probs_matrix.shape[1:])
    )

    segments = np.arange(0, instance_count).reshape(-1, 1)
    return (
        (top_verbs[segments, verb_ranks_idx], top_nouns[segments, noun_ranks_idx]),
        action_probs_matrix.reshape(instance_count, -1)[segments, action_ranks[:, :top_n]],
    )

def action_id_from_verb_noun(verb: Union[int, np.array], noun: Union[int, np.array]):
    """
    Examples:
    >>> action_id_from_verb_noun(0, 351)
    351
    >>> action_id_from_verb_noun(np.array([0, 1, 2]), np.array([0, 1, 2]))
    array([   0, 1001, 2002])
    """
    _ACTION_VERB_MULTIPLIER = 1000
    return verb * _ACTION_VERB_MULTIPLIER + noun

def convert_results(verb_scores, noun_scores) -> Dict[str, np.ndarray]:
    (verbs, nouns), scores = compute_action_scores(verb_scores, noun_scores, top_n=100) # verb ind, noun ind, action score = verb_scores[verb ind] * noun_scores[noun ind]
    action_results = [
        {
            action_id_from_verb_noun(verb, noun): score
            for verb, noun, score in zip(
                segment_verbs, segment_nouns, segment_score
            ) # per probability score
        }
        for segment_verbs, segment_nouns, segment_score in zip(verbs, nouns, scores) # per instance
    ]

    return action_results

def _scores_dict_to_ranks(scores: List[Dict[int, float]]) -> np.ndarray:
    """
    Compute ranking from class to score dictionary
    Examples:
        >>> _scores_dict_to_ranks([{0: 0.15, 10: 0.75, 5: 0.1},\
                                   {0: 0.85, 10: 0.10, 5: 0.05}])
        array([[10,  0,  5],
               [ 0, 10,  5]])
    """
    ranks = []
    for score in scores:
        class_ids = np.array(list(score.keys()))
        score_array = np.array([score[class_id] for class_id in class_ids])
        ranks.append(class_ids[np.argsort(score_array)[::-1]])
    return np.array(ranks)

def _scores_array_to_ranks(scores: np.ndarray):
    """
    The rank vector contains classes and is indexed by the rank
    Examples:
        >>> _scores_array_to_ranks(np.array([[0.1, 0.15, 0.25,  0.3, 0.5], \
                                             [0.5, 0.3, 0.25,  0.15, 0.1], \
                                             [0.2, 0.4,  0.1,  0.25, 0.05]]))
        array([[4, 3, 2, 1, 0],
               [0, 1, 2, 3, 4],
               [1, 3, 0, 2, 4]])
    """
    if scores.ndim != 2:
        raise ValueError(
            "Expected scores to be 2 dimensional: [n_instances, n_classes]"
        )
    return scores.argsort(axis=-1)[:, ::-1]

def selected_topk_accuracy(rankings, labels, ks, selected_class):
    if selected_class is not None:
        idx = labels == selected_class
        rankings = rankings[idx]
        labels = labels[idx]
    return topk_accuracy(rankings, labels, ks)

def topk_accuracy(
    rankings: np.ndarray, labels: np.ndarray, ks: Union[Tuple[int, ...], int] = (1, 5)
) -> List[float]:
    """Computes TOP-K accuracies for different values of k
    Parameters:
    -----------
    rankings
        2D rankings array: shape = (instance_count, label_count)
    labels
        1D correct labels array: shape = (instance_count,)
    ks
        The k values in top-k, either an int or a list of ints.
    Returns:
    --------
    list of float: TOP-K accuracy for each k in ks
    Raises:
    -------
    ValueError
         If the dimensionality of the rankings or labels is incorrect, or
         if the length of rankings and labels aren't equal
    """
    if isinstance(ks, int):
        ks = (ks,)

    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    accuracies = [tp[:, :k].max(1).mean() for k in ks]
    if any(np.isnan(accuracies)):
        raise ValueError(f"NaN present in accuracies {accuracies}")
    return accuracies

def mean_topk_recall(rankings, labels, k=5):
    classes = np.unique(labels)
    recalls = []
    for c in classes:
        recalls.append(selected_topk_accuracy(rankings, labels, ks=k, selected_class=c)[0])
    return np.mean(recalls)

def compute_action_recall(verb_scores, noun_scores, verb_gt, noun_gt, top_ks=(1, 5)):
    """
    verb_scores/noun_scores: np.ndarray of shape (instance_count, class_count) where each element is the predicted score 
    verb_gt/noun_gt: np.ndarray of shape (instance_count,) where each element is the gt class id (starting from 0)
    """
    # action
    action_class = action_id_from_verb_noun(verb_gt, noun_gt) # np.ndarray of shape (instance_count,) where each element is the gt class id
    scores = convert_results(verb_scores, noun_scores) # a list with len as #instances, each element is a dict with key as action id, value as score
    ranks = _scores_dict_to_ranks(scores) # array of shape (instance_count, top_k), each elememt is predicted action id
    
    recalls = []
    for top_k in top_ks: 
        recalls.append(mean_topk_recall(ranks, action_class, k=top_k))
    recalls = [item * 100 for item in recalls]

    # verb
    verb_ranks = _scores_array_to_ranks(verb_scores)
    verb_recalls = []
    for top_k in top_ks: 
        verb_recalls.append(mean_topk_recall(verb_ranks, verb_gt, k=top_k))
    verb_recalls = [item * 100 for item in verb_recalls]

    # noun
    noun_ranks = _scores_array_to_ranks(noun_scores)
    noun_recalls = []
    for top_k in top_ks: 
        noun_recalls.append(mean_topk_recall(noun_ranks, noun_gt, k=top_k))
    noun_recalls = [item * 100 for item in noun_recalls]

    print(recalls, verb_recalls, noun_recalls)
    
    return recalls, verb_recalls, noun_recalls

