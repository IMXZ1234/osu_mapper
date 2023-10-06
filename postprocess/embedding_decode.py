import itertools
import pickle
import numpy as np


class EmbeddingDecode:
    def __init__(self, embedding_path, beat_divisor=8, subset_path=None):
        """
        subset_path: contains a list of label_idxs available for decoding
        """
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)

        label_idx_to_beat_label_seq = np.array(list(itertools.product(*[(0, 1, 2, 3) for _ in range(beat_divisor)])))

        if subset_path is not None:
            with open(subset_path, 'rb') as f:
                embedding_counter = pickle.load(f)
            subset = np.array(list(embedding_counter.keys()))
        else:
            subset = np.arange(len(embedding))

        # N, C
        self.considered_embedding = embedding[subset]
        self.considered_embedding_normed = self.considered_embedding / np.linalg.norm(self.considered_embedding, axis=1, keepdims=True)
        self.considered_beat_label_seq = label_idx_to_beat_label_seq[subset]

    def decode(self, embedding_output):
        """
        embedding_output: L, C

        returns decoded snap level label seq
        """
        embedding_output_normed = embedding_output / np.linalg.norm(embedding_output, axis=1, keepdims=True)
        similarity = np.matmul(
            embedding_output_normed, self.considered_embedding_normed.T
        )
        decoded_label_idx = np.argmax(similarity, axis=1)
        decoded = self.considered_beat_label_seq[decoded_label_idx].reshape(-1)
        return decoded
