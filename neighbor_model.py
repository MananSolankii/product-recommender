"""Lesson 7: keep the strongest outgoing item-similarity links."""
from numbers import Integral
import numpy as np
from scipy.sparse import csr_matrix
from collaborative_model import CollaborativeRecommender


def keep_top_neighbors(similarity, neighbors):
    """Return a new CSR matrix with <= neighbors positive links per row."""
    if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or neighbors < 1:
        raise ValueError("neighbors must be a positive integer.")
    source = similarity.tocsr(copy=True)
    source.sum_duplicates()
    source.setdiag(0)
    source.eliminate_zeros()
    data, indices, indptr = [], [], [0]
    for row in range(source.shape[0]):
        start, stop = source.indptr[row:row + 2]
        columns = source.indices[start:stop]
        values = source.data[start:stop]
        positive = values > 0
        columns, values = columns[positive], values[positive]
        # Primary: descending similarity. Ties: existing popularity index.
        order = np.lexsort((columns, -values))[:neighbors]
        indices.extend(columns[order].tolist())
        data.extend(values[order].tolist())
        indptr.append(len(data))
    result = csr_matrix((data, indices, indptr), shape=source.shape, dtype=float)
    result.sort_indices()
    return result


class NeighborRecommender(CollaborativeRecommender):
    def __init__(self, history, neighbors=30):
        # Validate before the potentially expensive base fit.
        if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or neighbors < 1:
            raise ValueError("neighbors must be a positive integer.")
        super().__init__(history)
        before = self.similarity.nnz
        self.similarity = keep_top_neighbors(self.similarity, neighbors)
        self.neighbors = int(neighbors)
        self.neighbor_summary = {
            "neighbors_per_item": self.neighbors,
            "directed_links_before": int(before),
            "directed_links_after": int(self.similarity.nnz),
        }
        # Inherit scores_for and recommend: sum outgoing links from liked items.
        # Row pruning can make this matrix asymmetric. Do not symmetrize it.
