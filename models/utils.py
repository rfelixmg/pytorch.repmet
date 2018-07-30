import numpy as np
from data.utils import get_inputs

def compute_reps(net, dataset, indexs, chunk_size=None):
    """Compute representations/embeddings

    :param net: The net to foward through
    :param dataset: the dataset
    :param indexs: the indexs of the samples to pass
    :return: embeddings as a numpy array
    """
    if chunk_size:
        initial_reps = []
        for s in range(0, len(indexs), chunk_size):
            indexs_inner = list(indexs[s:min(s + chunk_size, len(indexs))])
            initial_reps.append(net(get_inputs(dataset, indexs_inner)).detach().cpu().numpy())

        return np.vstack(initial_reps)
    else:
        return net(get_inputs(dataset, indexs)).detach().cpu().numpy()
        # return net(get_inputs(dataset, indexs)*255).detach().cpu().numpy()  # MNIST only learns when is 0-255 not 0-1

def compute_all_reps(net, dataset, chunk_size):
    """Compute representations for entire set in chunks (sequential non-shuffled batches).

    Basically just forwards the inputs through the net to get the embedding vectors in 'chunks'
    """
    return compute_reps(net, dataset, list(range(len(dataset))), chunk_size=chunk_size)