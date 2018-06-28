import networkx as nx


def slice_graph(graph, root=None, subgraph=()):
    if not root:
        root = next(nx.topological_sort(graph))
    if not subgraph:
        subgraph = (root,)
    else:
        subgraph = (*subgraph, root)

    children = list(graph.successors(root))
    if len(children) == 0:
        # add the subgraph to the final list
        return subgraph
    all_subgraphs = []
    for child in children:
        # recursively go through the children
        subgraphs_from_child = slice_graph(graph, root=child)
        candidates = []
        for sub_g in subgraphs_from_child:
            candidates.append((root, *sub_g) if isinstance(sub_g, tuple) else (root, sub_g))
        all_subgraphs.extend(candidates)
    return all_subgraphs

