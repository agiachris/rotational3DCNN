import numpy as np
import torch
import torch.utils.data

class Metric:
  # takes Height x Width x Length tensor filled with 0/1
  def get_accuracy(pred, labels):
      # gets voxel-wise accuracy per batch
      correctness = torch.eq(pred, labels)
      accuracy = torch.sum(correctness).item()/pred.nelement()
      return accuracy

  # takes Batch x Height x Width x Length tensor filled with 0/1
  def get_mean_accuracy(pred, labels):
    # gets accuracy for all data
    accuracy = [get_accuracy(pred[i], labels[i]) for i in range(len(pred))]
    return np.array(accuracy).mean()

  # takes Height x Width x Length tensor filled with 0/1
  def get_iou(pred, labels):
      # gets intersection over union for a signle batch
      pred = pred.squeeze(1)
      labels = labels.squeeze(1)
      intersection = np.logical_and(labels, pred)
      union = np.logical_or(labels, pred)
      iou = torch.sum(intersection) / torch.sum(union)
      return iou
  
  # takes Batch x Height x Width x Length tensor filled with 0/1
  def get_mean_iou(pred, labels):
    # gets intersection over union for all data
    iou = [get_iou(pred[i], labels[i]) for i in range(len(pred))]
    return np.array(iou).mean()