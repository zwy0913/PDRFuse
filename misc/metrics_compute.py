import torch.nn.functional as F
import torchvision.transforms.v2.functional as tv2f


# Compute common loss and metric for generator only
def compute_loss_and_metrics_focus(images, has_lower_res=True):

    """
    This part compute loss and metrics for the generator
    """

    loss_and_metrics = {}

    gt = images['gt']
    # seg = images['seg']

    pred_224 = images['pred_224']
    pred_28_3 = images['pred_28']
    pred_56_2 = images['pred_56']

    # Loss weights
    ce_weights = [0.0, 1.0, 1.0]
    l1_weights = [1.0, 0.0, 0.5]
    l2_weights = [1.0, 0.0, 0.5]

    # temp holder for losses at different scale
    ce_loss = [0., 0., 0.]
    l1_loss = [0., 0., 0.]
    l2_loss = [0., 0., 0.]
    loss = [0., 0., 0.]

    ce_loss[0] = F.binary_cross_entropy_with_logits(images['out_224'], (gt > 0.5).float())
    l1_loss[0] = F.l1_loss(pred_224, gt)
    l2_loss[0] = F.mse_loss(pred_224, gt)
    ce_loss[1] = F.binary_cross_entropy_with_logits(images['out_28'], (tv2f.resize(gt, [images['out_28'].shape[2], images['out_28'].shape[3]], antialias=False) > 0.5).float())
    l1_loss[1] = F.l1_loss(pred_28_3, tv2f.resize(gt, [pred_28_3.shape[2], pred_28_3.shape[3]], antialias=False))
    l2_loss[1] = F.mse_loss(pred_28_3, tv2f.resize(gt, [pred_28_3.shape[2], pred_28_3.shape[3]], antialias=False))
    ce_loss[2] = F.binary_cross_entropy_with_logits(images['out_56'], (tv2f.resize(gt, [images['out_56'].shape[2], images['out_56'].shape[3]], antialias=False) > 0.5).float())
    l1_loss[2] = F.l1_loss(pred_56_2, tv2f.resize(gt, [pred_56_2.shape[2], pred_56_2.shape[3]], antialias=False))
    l2_loss[2] = F.mse_loss(pred_56_2, tv2f.resize(gt, [pred_56_2.shape[2], pred_56_2.shape[3]], antialias=False))

    loss_and_metrics['grad_loss'] = F.l1_loss(images['pred_sobel'], images['gt_sobel'])

    # Weighted loss for different levels
    for i in range(3):
        loss[i] = ce_loss[i] * ce_weights[i] + \
                l1_loss[i] * l1_weights[i] + \
                l2_loss[i] * l2_weights[i]

    loss[0] += loss_and_metrics['grad_loss'] * 5


    """
    All done.
    Now gather everything in a dict for logging
    """

    loss_and_metrics['total_loss'] = 0
    for i in range(3):
        loss_and_metrics['ce_loss/s_%d' % i] = ce_loss[i]
        loss_and_metrics['l1_loss/s_%d' % i] = l1_loss[i]
        loss_and_metrics['l2_loss/s_%d' % i] = l2_loss[i]
        loss_and_metrics['loss/s_%d' % i] = loss[i]

        loss_and_metrics['total_loss'] += loss[i]

    return loss_and_metrics


