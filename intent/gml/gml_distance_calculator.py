import numpy as np
from scipy.spatial.distance import cdist

from intent.embed.transform import doc2vec
from intent.gml.gml_slicer import find_root


def _reshape_1d_vec_2d(vec):
    """
    Transforms 1D vector of length N to the matrix (N,1)
    :param vec: vector to transform
    :return: reshaped vector
    """
    return vec.reshape(1, -1)


def extract_single_value(val, penalty=1000):
    """
    Scipy distance returns 2-dimensional array, returning the single value
    :param penalty: value to be taken as distance if nan
    :param val: 2D nd.array with single value
    :return: single value
    """
    res = val.flatten()[0]
    return penalty if np.isnan(res) else np.abs(res)


def calculate_cosine(vec1, vec2):
    return cdist(vec1, vec2, metric='cosine')


def calculate_doc2vec(graph, vec_size, ft_model):
    for node in graph:
        graph.node[node]['text'] = doc2vec(graph.node[node]['text'], vec_size, ft_model)
        if graph.node[node]['text'].max() == 0. and graph.node[node]['text'].min() == 0:
            print('[WARNING] Suspiciously empty document vector: only zeros')


def calculate_distance(graph, distance_lambda=calculate_cosine):
    root = find_root(graph)
    post_vec = _reshape_1d_vec_2d(graph.nodes[root]['text'])
    for node in graph:
        node_vec = _reshape_1d_vec_2d(graph.node[node]['text'])
        graph.node[node]['distanceRoot'] = extract_single_value(distance_lambda(node_vec, post_vec))
        parents = list(graph.predecessors(node))
        if len(parents) == 0:
            # this is the root node
            graph.node[node]['distanceParent'] = 0
            continue
        for p in graph.predecessors(node):
            parent_vec = _reshape_1d_vec_2d(graph.node[p]['text'])
            graph.node[node]['distanceParent'] = extract_single_value(distance_lambda(node_vec, parent_vec))
