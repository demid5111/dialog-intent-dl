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
        'isRoot': False,
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


def create_graph_df(graph, node_ids=None, only_distance=False):
    mandatory_header = [
        'ID of comment',
        'ID of post',
        'Likes',
        'Intent analysis',
        'Content analysis',
        'Distance to parent',
        'Distance to post'
    ]
    optional_header = [
        'Doc2Vec value',
    ]

    header = []
    header.extend(mandatory_header)
    if not only_distance:
        header.extend(optional_header)

    csv_name_to_prop = {
        'ID of comment': 'commentID',
        'ID of post': 'postID',
        'Likes': 'likes',
        'Intent analysis': 'intentLabels',
        'Content analysis': 'contentLabels',
        'Doc2Vec value': 'text',
        'Distance to parent': 'distanceParent',
        'Distance to post': 'distanceRoot',
    }

    lines = []
    if not node_ids:
        node_ids = traverse_sequence(graph)
    for child in node_ids:
        tmp = []
        for i in header:
            val = graph.node[child][csv_name_to_prop[i]]
            if csv_name_to_prop[i] == 'text':
                val = np.array2string(val, formatter={'float_kind': lambda x: "%.6f" % x})
            tmp.append(val)
        lines.append(tmp)

    return pd.DataFrame(data=np.array(lines), columns=header)


def dump_graph_csv(graph, file_name='output/output', node_ids=None, only_distance=False, output_format='csv'):
    df = create_graph_df(graph, node_ids, only_distance=only_distance)

    if output_format == 'hdf5':
        df.to_hdf('{}.h5'.format(file_name), key='data')
    else:
        df.to_csv('{}.csv'.format(file_name))

    return df


def transform_chains_from_root(dfs):
    return [transform_chain_to_row(df, key='Distance to post') for df in dfs]


def transform_chains_from_neighbor(dfs):
    return [transform_chain_to_row(df, key='Distance to parent') for df in dfs]


def transform_chain_to_row(df, key):
    only_vals = df[[key]]
    return [row[key] for idx, row in only_vals.iterrows()]


def transform_chains_to_rows(dfs):
    return transform_chains_from_root(dfs), transform_chains_from_neighbor(dfs)


def vectors_to_df(vecs):
    df = pd.DataFrame(data=vecs)
    return df


def create_df_from(vecs):
    return vectors_to_df(vecs)


def dump_vectors_to_excel(df1, df2, file_name='output/distances.xlsx', df1_name='From Root', df2_name='From Parent'):
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

    df1.to_excel(writer, sheet_name=df1_name, index=False, header=False)
    df2.to_excel(writer, sheet_name=df2_name, index=False, header=False)

    writer.save()
