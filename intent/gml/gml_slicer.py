import networkx as nx

from intent.gml.gml_utils import dump_graph_for_graphviz


def slice_graph_recursive(graph, root=None, subgraph=()):
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
        subgraphs_from_child = slice_graph_recursive(graph, root=child)
        candidates = []
        for sub_g in subgraphs_from_child:
            candidates.append((root, *sub_g) if isinstance(sub_g, tuple) else (root, sub_g))
        all_subgraphs.extend(candidates)
    return all_subgraphs


def slice_graph_nx(graph):
    root = next(nx.topological_sort(graph))

    paths = []
    for node in graph:
        if graph.out_degree(node) == 0:  # it's a leaf
            paths.extend(list(nx.all_simple_paths(graph, root, node)))
    return paths


def clean_up_graph(graph):
    empty_nodes = []

    for node in graph:
        if graph.out_degree(node) == 0 and graph.in_degree(node) == 0:  # it's a leaf
            empty_nodes.append(node)

    if empty_nodes:
        print('Found empty nodes. Removing...')
        graph.remove_nodes_from(empty_nodes)


def connect_missed_subgraphs(graph):
    # find the root (the node with maximum output nodes and 0 inputs)
    max_degree = -1
    root = None
    pseudo_roots = []
    for node in graph:
        if graph.in_degree(node) == 0 and graph.out_degree(node) > 0:
            pseudo_roots.append(node)

        if graph.in_degree(node) == 0 and graph.out_degree(node) > max_degree:
            max_degree = graph.out_degree(node)
            root = node

    graph.add_edges_from(((root, i) for i in pseudo_roots if i != root))
