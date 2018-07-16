import networkx as nx
import re

from intent.post.post_reader import get_posts_col, get_text_col


def find_root(graph):
    root = None
    for node in graph:
        if graph.nodes[node]['isRoot']:
            root = node
            break
    return root


def slice_graph_recursive(graph, root=None, subgraph=()):
    if not root:
        root = find_root(graph)

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
    root = find_root(graph)

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
    root = find_root(graph)
    pseudo_roots = []
    for node in graph:
        if graph.in_degree(node) == 0 and graph.out_degree(node) > 0:
            pseudo_roots.append(node)

    graph.add_edges_from(((root, i) for i in pseudo_roots if i != root))


def _filter_extra_intent_path(graph, path):
    candidates = [graph.nodes[el_id] for el_id in path if re.match('^\d+0(\d{2}|0\d)$', el_id)]
    return bool([i for i in candidates if i['commentID'][:-3] in graph.nodes])


def remove_extra_intentions(graph):
    """
    The graph contains such paths:
        node1113 (main intent)
        /
    root
        \
        node1113001 (auxiliary intent)

    Removing all such nodes with suffix 00(1,2,3,...) with all the paths after them
    :param graph: the original graph
    :param paths: list of paths
    """
    root = find_root(graph)

    to_remove = []
    for node in graph:
        el_id = graph.nodes[node]['commentID']
        if re.match('^\d+0(\d{2}|0\d)$', el_id) and el_id[:-3] in graph.nodes:
            to_remove.append(node)
    graph.remove_nodes_from(to_remove)

    i = 0
    for comp in nx.connected_components(graph.to_undirected()):
        if root not in comp:
            i += 1
            graph.remove_nodes_from(comp)

    if i > 0:
        print('[INFO] Removed {} connected components that does not contain main root'.format(i))


def embed_post_text(graph, df):
    root = find_root(graph)
    postID = graph.nodes[root]['postID']
    df_tmp = df.loc[get_posts_col(df) == int(postID)]
    if df_tmp.shape[0] == 0:
        raise ValueError('Unable to find post id {}'.format(postID))
    graph.nodes[root]['text'] = get_text_col(df_tmp.iloc[0])
