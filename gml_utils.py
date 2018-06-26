import copy
import networkx as nx
import pandas as pd
import numpy as np


def dump_graph_for_graphviz(graph: nx.MultiDiGraph, node_attrs: list = ['kind', 'op', 'shape'],
                            edge_attrs: list = ['in', 'out']):
    nodes_to_dump = graph.nodes()
    string = '\ndigraph {\n'
    for src_node_name, dst_node_name, attrs in graph.edges(data=True):
        if src_node_name not in nodes_to_dump or dst_node_name not in nodes_to_dump:
            continue
        src_node = graph.node[src_node_name]
        dst_node = graph.node[dst_node_name]
        src_node_string = str(src_node_name) + '\\n' + '\\n'.join(
            [str(key) + '=' + str(src_node.get(key, 'None')) for key in node_attrs if key in src_node])
        dst_node_string = str(dst_node_name) + '\\n' + '\\n'.join(
            [str(key) + '=' + str(dst_node.get(key, 'None')) for key in node_attrs if key in dst_node])
        edge_string = ' '.join([str(key) + '=' + str(attrs.get(key, 'None')) for key in edge_attrs if key in attrs])
        string += '"{}" -> "{}" [label = "{}"];\n'.format(src_node_string, dst_node_string, edge_string)
    string += '}'
    return string


def new_attrs():
    return copy.deepcopy({
        'postID': None,
        'commentID': None,
        'text': None,
        'likes': None,
        'intentLabels': None,
        'contentLabels': None,
    })


def traverse_sequence(graph, root=None, seq=()):
    if not root:
        root = next(nx.topological_sort(graph))
    seq = (*seq, root)
    children = list(graph.successors(root))
    if len(children) > 1:
        raise NotImplementedError('traverse_sequence function works only for sequences')
    if len(children) == 0:
        return seq
    return traverse_sequence(graph, root=children[0], seq=seq)


def create_graph_df(graph):
    header = ['ID of comment', 'ID of post', 'Likes', 'Intent analysis', 'Content analysis', 'Text']
    csv_name_to_prop = {
        'ID of comment': 'commentID',
        'ID of post': 'postID',
        'Likes': 'likes',
        'Intent analysis': 'intentLabels',
        'Content analysis': 'contentLabels',
        'Text': 'text'
    }
    lines = []
    node_ids = traverse_sequence(graph)
    for child in node_ids:
        lines.append([graph.node[child][csv_name_to_prop[i]] for i in header])

    return pd.DataFrame(data=np.array(lines), columns=header)


def dump_graph_csv(graph, file_name='output/output'):
    df = create_graph_df(graph)

    df.to_csv('{}.csv'.format(file_name))
