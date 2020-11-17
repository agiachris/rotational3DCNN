import torch
import numpy as np

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

    def __init__(self, config):
        """Metric computation class capable of computing IoU scores, accuracy scores, and l2-distance
         scores for a batch of signed-distance field and distance field input-label pairs.
        """
        self.batch_size = int(config['batch_size'])

    # takes Height x Width x Length tensor filled with 0/1
    def get_accuracy_per_object(self, pred, labels):
        # gets voxel-wise accuracy a single object
        correctness = torch.eq(pred, labels)
        accuracy = torch.sum(correctness).item() / pred.nelement()
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
        # gets intersection over union for a single object
        intersection = np.logical_and(labels, pred)
        union = np.logical_or(labels, pred)
        iou = torch.sum(intersection).item() / torch.sum(union).item()
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

    def l2_norm(self, pred, labels):
        """Compute the L2 norm between predictions and labels
        """
        pred = pred.reshape(self.batch_size, -1)
        labels = labels.reshape(self.batch_size, -1)
        assert (pred.size() == labels.size())
        return torch.norm(pred - labels) / pred.nelement()
