import copy
import networkx as nx

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