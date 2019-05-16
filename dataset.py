from config import *
import torch
import numpy as np
import data_import as data

device = "cuda:0" if CUDA and torch.cuda.is_available() else "cpu"


class Dataset:
    """
        Hides the different caluclations required for if using the proper dataset vs the dummy. So the same methods
        can be called for both when used by the developer
    """

    def __init__(self):
        if not USE_D1:
            self.processor = data.DataProcessing()
            self.training_data_iterator = self.get_training_data_iterator()

    def get_w_emb_matrix(self):
        # returns the trained glove embedded vectors concatenated as a matrix [118646, 50]
        if USE_D1:
            return torch.Tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1]])
        else:
            return self.processor.text.vocab.vectors

    def get_training_data_str(self):
        return np.array([["i", "am", "potato"], ["big", "brown", "potato"], ["am", "i", "brown"]])

    def get_vocabulary(self):
        vocabulary = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        return vocabulary

    def get_training_data_indices(self):
        return torch.tensor([[0, 1, 2, 4], [3, 4, 2, 4], [1, 0, 4, 4]], dtype=torch.long).to(
            device)

    def get_training_data(self, training_data_indices=None):
        if USE_D1:
            training_data = torch.Tensor([[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
                                          [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
                                          [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]]).to(
                device)
        else:
            training_data = self.processor.data_to_embeddings(training_data_indices, self.processor.text)
        return training_data

    def get_gt_summaries(self, gt_summaries_indices=None):
        if USE_D1:
            gt_summaries = torch.Tensor([[[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
                                         [[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
                                         [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]]])
        else:
            gt_summaries = self.processor.data_to_embeddings(gt_summaries_indices, self.processor.text)
        return gt_summaries

    def get_gt_summaries_indices(self):
        gt_summaries_indices = torch.tensor([[0, 2, 4], [4, 2, 4], [4, 4, 2]])
        return gt_summaries_indices

    def get_processor(self):
        return self.processor

    def get_indices(self):
        # get training_data and gt_summary indicies (in that order!)
        if USE_D1:
            return self.get_training_data_indices(), self.get_gt_summaries_indices()
        else:
            return next(self.training_data_iterator)

    def get_training_data_iterator(self):
        return iter(self.processor.train_data)

    def get_embedding_size(self):
        if USE_D1:
            embedding_size = 5
        else:
            embedding_size = self.processor.embedding_size
        return embedding_size

    def get_hidden_size(self):
        hidden_size = 7 if USE_D1 else 200
        return hidden_size
