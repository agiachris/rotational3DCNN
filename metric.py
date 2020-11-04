import numpy as np
import torch

'''
example usage 

data = [[[[0, 0],[0, 0]],[[0, 1],[1, 1]]], [[[1, 1],[0, 1]],[[0, 0],[1, 1]]]]
data_tensor = torch.FloatTensor(data)
label = [[[[1, 0],[0, 1]],[[0, 1],[1, 1]]], [[[0, 1],[1, 0]],[[1, 1],[1, 1]]]]
label_tensor = torch.FloatTensor(label)
metric = Metric()
print(metric.get_iou_per_batch(data_tensor, label_tensor))
print(metric.get_accuracy_per_batch(data_tensor, label_tensor))
'''

class Metric:
  # takes Height x Width x Length tensor filled with 0/1
  def get_accuracy_per_object(self, pred, labels):
      # gets voxel-wise accuracy a signle object
      correctness = torch.eq(pred, labels)
      accuracy = torch.sum(correctness).item()/pred.nelement()
      return accuracy

  # takes Batch x Height x Width x Length tensor/array filled with 0/1
  def get_accuracy_per_batch(self, pred, labels):
    # gets accuracy for a batch
    accuracy = [self.get_accuracy_per_object(pred[i], labels[i]) for i in range(len(pred))]
    return np.array(accuracy).mean()

  # takes Number_of_batches x Batch x Height x Width x Length tensor/array filled with 0/1
  def get_accuracy_for_all_data(self, pred, labels):
    # gets intersection over union for all data
    iou = [self.get_accuracy_per_batch(pred[i], labels[i]) for i in range(len(pred))]
    return np.array(iou).mean()

  # takes Height x Width x Length tensor filled with 0/1
  def get_iou_per_object(self, pred, labels):
      # gets intersection over union for a signle object
      pred = pred.squeeze(1)
      labels = labels.squeeze(1)
      intersection = np.logical_and(labels, pred)
      union = np.logical_or(labels, pred)
      iou = torch.sum(intersection) / torch.sum(union)
      return iou
  
  # takes Batch x Height x Width x Length tensor/array filled with 0/1
  def get_iou_per_batch(self, pred, labels):
    # gets intersection over union for a batch
    iou = [self.get_iou_per_object(pred[i], labels[i]) for i in range(len(pred))]
    return np.array(iou).mean()

  # takes Number_of_batches x Batch x Height x Width x Length tensor/array filled with 0/1
  def get_iou_for_all_data(self, pred, labels):
    # gets intersection over union for all data
    iou = [self.get_iou_per_batch(pred[i], labels[i]) for i in range(len(pred))]
    return np.array(iou).mean()
