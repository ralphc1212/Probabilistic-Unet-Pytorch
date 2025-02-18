import torch
import numpy as np

def calc_confusion(labels, samples, class_ixs, loss_mask=None):
    """
    Compute confusion matrix for each class across the given arrays.
    Assumes classes are given in integer-valued encoding.
    :param labels: 4/5D array
    :param samples: 4/5D array
    :param class_ixs: integer or list of integers specifying the classes to evaluate
    :param loss_mask: 4/5D array
    :return: 2D array
    """
    try:
        assert labels.shape == samples.shape
    except:
        raise AssertionError('shape mismatch {} vs. {}'.format(labels.shape, samples.shape))

    if isinstance(class_ixs, int):
        num_classes = class_ixs
        class_ixs = range(class_ixs)
    elif isinstance(class_ixs, list):
        num_classes = len(class_ixs)
    else:
        raise TypeError('arg class_ixs needs to be int or list, not {}.'.format(type(class_ixs)))

    if loss_mask is None:
        # shp = labels.shape
        # loss_mask = np.zeros(shape=(shp[0], 1, shp[2], shp[3]))
        loss_mask = np.zeros_like(labels)

    conf_matrix = np.zeros(shape=(num_classes, 4), dtype=np.float32)
    for i,c in enumerate(class_ixs):

        pred_ = (samples == c).astype(np.uint8)
        labels_ = (labels == c).astype(np.uint8)

        conf_matrix[i, 0] = int(((pred_ != 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # TP
        conf_matrix[i, 1] = int(((pred_ != 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # FP
        conf_matrix[i, 2] = int(((pred_ == 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # TN
        conf_matrix[i, 3] = int(((pred_ == 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # FN

    return conf_matrix


def get_energy_distance_components(gt_seg_modes, seg_samples, eval_class_ids, ignore_mask=None):
    """
    Calculates the components for the IoU-based generalized energy distance given an array holding all segmentation
    modes and an array holding all sampled segmentations.
    :param gt_seg_modes: N-D array in format (num_modes,[...],H,W)
    :param seg_samples: N-D array in format (num_samples,[...],H,W)
    :param eval_class_ids: integer or list of integers specifying the classes to encode, if integer range() is applied
    :param ignore_mask: N-D array in format ([...],H,W)
    :return: dict
    """
    num_modes = gt_seg_modes.shape[0]
    num_samples = seg_samples.shape[0]

    if isinstance(eval_class_ids, int):
        eval_class_ids = list(range(eval_class_ids))

    d_matrix_YS = np.zeros(shape=(num_modes, num_samples, len(eval_class_ids)), dtype=np.float32)
    d_matrix_YY = np.zeros(shape=(num_modes, num_modes, len(eval_class_ids)), dtype=np.float32)
    d_matrix_SS = np.zeros(shape=(num_samples, num_samples, len(eval_class_ids)), dtype=np.float32)

    # iterate all ground-truth modes
    for mode in range(num_modes):

        ##########################################
        #   Calculate d(Y,S) = [1 - IoU(Y,S)],	 #
        #   with S ~ P_pred, Y ~ P_gt  			 #
        ##########################################

        # iterate the samples S
        for i in range(num_samples):
            conf_matrix = calc_confusion(gt_seg_modes[mode], seg_samples[i],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_YS[mode, i] = 1. - iou

        ###########################################
        #   Calculate d(Y,Y') = [1 - IoU(Y,Y')],  #
        #   with Y,Y' ~ P_gt  	   				  #
        ###########################################

        # iterate the ground-truth modes Y' while exploiting the pair-wise symmetries for efficiency
        for mode_2 in range(mode, num_modes):
            conf_matrix = calc_confusion(gt_seg_modes[mode], gt_seg_modes[mode_2],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_YY[mode, mode_2] = 1. - iou
            d_matrix_YY[mode_2, mode] = 1. - iou

    #########################################
    #   Calculate d(S,S') = 1 - IoU(S,S'),  #
    #   with S,S' ~ P_pred        			#
    #########################################

    # iterate all samples S
    for i in range(num_samples):
        # iterate all samples S'
        for j in range(i, num_samples):
            conf_matrix = calc_confusion(seg_samples[i], seg_samples[j],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_SS[i, j] = 1. - iou
            d_matrix_SS[j, i] = 1. - iou

    return {'YS': d_matrix_YS, 'SS': d_matrix_SS, 'YY': d_matrix_YY}


def calc_energy_distances(d_matrices, num_samples=None, probability_weighted=False, label_switches=None, exp_mode=5):
    """
    Calculate the energy distance for each image based on matrices holding the combinatorial distances.
    :param d_matrices: dict holding 4D arrays of shape \
    (num_images, num_modes/num_samples, num_modes/num_samples, num_classes)
    :param num_samples: integer or None
    :param probability_weighted: bool
    :param label_switches: None or dict
    :param exp_mode: integer
    :return: numpy array
    """
    d_matrices = d_matrices.copy()

    if num_samples is None:
        num_samples = d_matrices['SS'].shape[1]

    d_matrices['YS'] = d_matrices['YS'][:,:,:num_samples]
    d_matrices['SS'] = d_matrices['SS'][:,:num_samples,:num_samples]

    # perform a nanmean over the class axis so as to not factor in classes that are not present in
    # both the ground-truth mode as well as the sampled prediction
    if probability_weighted:
       mode_stats = get_mode_statistics(label_switches, exp_modes=exp_mode)
       mode_probs = mode_stats['mode_probs']

       mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
       mean_d_YS = np.mean(mean_d_YS, axis=2)
       mean_d_YS = mean_d_YS * mode_probs[np.newaxis, :]
       d_YS = np.sum(mean_d_YS, axis=1)

       mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
       d_SS = np.mean(mean_d_SS, axis=(1, 2))

       mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
       mean_d_YY = mean_d_YY * mode_probs[np.newaxis, :, np.newaxis] * mode_probs[np.newaxis, np.newaxis, :]
       d_YY = np.sum(mean_d_YY, axis=(1, 2))

    else:

       mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
       d_YS = np.mean(mean_d_YS, axis=(1,2))

       mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
       d_SS = np.mean(mean_d_SS, axis=(1, 2))

       mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
       d_YY = np.nanmean(mean_d_YY, axis=(1, 2))

    return 2 * d_YS - d_SS - d_YY


def metrics_from_conf_matrix(conf_matrix):
    """
    Calculate IoU per class from a confusion_matrix.
    :param conf_matrix: 2D array of shape (num_classes, 4)
    :return: dict holding 1D-vectors of metrics
    """
    tps = conf_matrix[:,0]
    fps = conf_matrix[:,1]
    fns = conf_matrix[:,3]

    metrics = {}
    metrics['iou'] = np.zeros_like(tps, dtype=np.float32)

    # iterate classes
    for c in range(tps.shape[0]):
        # unless both the prediction and the ground-truth is empty, calculate a finite IoU
        if tps[c] + fps[c] + fns[c] != 0:
            metrics['iou'][c] = tps[c] / (tps[c] + fps[c] + fns[c])
        else:
            metrics['iou'][c] = np.nan

    return metrics


def get_mode_statistics(label_switches, exp_modes=5):
    """
    Calculate a binary matrix of switches as well as a vector of mode probabilities.
    :param label_switches: dict specifying class names and their individual sampling probabilities
    :param exp_modes: integer, number of independently switchable classes
    :return: dict
    """
    num_modes = 2 ** exp_modes

    # assemble a binary matrix of switch decisions
    switch = np.zeros(shape=(num_modes, 5), dtype=np.uint8)
    for i in range(exp_modes):
        switch[:,i] = 2 ** i * (2 ** (exp_modes - 1 - i) * [0] + 2 ** (exp_modes - 1 - i) * [1])

    # calculate the probability for each individual mode
    mode_probs = np.zeros(shape=(num_modes,), dtype=np.float32)
    for mode in range(num_modes):
        prob = 1.
        for i, c in enumerate(label_switches.keys()):
            if switch[mode, i]:
                prob *= label_switches[c]
            else:
                prob *= 1. - label_switches[c]
        mode_probs[mode] = prob
    assert np.sum(mode_probs) == 1.

    return {'switch': switch, 'mode_probs': mode_probs}

