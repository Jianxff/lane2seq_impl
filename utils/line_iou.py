import numpy as np

# def line_iou(pred, target, img_w, length=15, aligned=True):
#     '''
#     Calculate the line iou value between predictions and targets
#     Args:
#         pred: lane predictions, shape: (num_pred, 72)
#         target: ground truth, shape: (num_target, 72)
#         img_w: image width
#         length: extended radius
#         aligned: True for iou loss calculation, False for pair-wise ious in assign
#     '''
#     px1 = pred - length
#     px2 = pred + length
#     tx1 = target - length
#     tx2 = target + length
#     if aligned:
#         invalid_mask = target 
#         ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
#         union = torch.max(px2, tx2) - torch.min(px1, tx1)
#     else:
#         num_pred = pred.shape[0]
#         invalid_mask = target.repeat(num_pred, 1, 1)
#         ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
#                torch.max(px1[:, None, :], tx1[None, ...]))
#         union = (torch.max(px2[:, None, :], tx2[None, ...]) -
#                  torch.min(px1[:, None, :], tx1[None, ...]))

#     invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
#     ovr[invalid_masks] = 0.
#     union[invalid_masks] = 0.
#     iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
#     return iou


def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 50)
        target: ground truth, shape: (num_target, 50)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target 
        ovr = np.minimum(px2, tx2) - np.maximum(px1, tx1)
        union = np.maximum(px2, tx2) - np.minimum(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = np.repeat(target, num_pred, axis=0)
        ovr = (np.minimum(px2[:, None, :], tx2[None, ...]) -
               np.maximum(px1[:, None, :], tx1[None, ...]))
        union = (np.maximum(px2[:, None, :], tx2[None, ...]) -
                 np.minimum(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(axis=-1) / (union.sum(axis=-1) + 1e-9)
    return iou