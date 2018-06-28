from intent.embed.transform import doc2vec


def calculate_doc2vec_l2_norm(graph, vec_size, ft_model):
    for node in graph:
        graph.node[node]['text'] = doc2vec(graph.node[node]['text'], vec_size, ft_model)