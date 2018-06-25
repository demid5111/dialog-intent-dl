import argparse
import os
import networkx as nx
import re
import copy


def unique_id(graph: nx.MultiDiGraph, prefix: str = ""):
    unique_id.count = max(unique_id.count, graph.number_of_nodes()) + 1
    return prefix + str(unique_id.count)


unique_id.count = 0


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


def load_graph(graph_path):
    g = nx.MultiDiGraph()
    with open(graph_path) as graph_file:
        is_in_node = False
        is_in_edge = False
        is_post_node = True
        from_node = None
        to_node = None
        post_id = None
        attrs = new_attrs()
        for line in graph_file.readlines():
            if re.match('^\s*node\n$', line):
                if is_in_node:
                    new_node_attrs = copy.deepcopy(attrs)
                    g.add_node(unique_id(g), **new_node_attrs)
                    attrs = new_attrs()
                is_in_node = True
                continue
            if re.match('^\s*edge\n$', line):
                if is_in_edge:
                    g.add_edge(from_node, to_node)
                    from_node = None
                    to_node = None
                is_in_edge = True
                continue
            if is_in_node and re.match('^\s*id\s\d+\n$', line):
                # TODO: if it is first - it is post ID
                el_id = re.split('d+', line)[-1].strip()
                if is_post_node:
                    attrs['postID'] = el_id
                    post_id = attrs['postID']
                else:
                    attrs['postID'] = post_id
                    attrs['commentID'] = el_id
            elif is_in_node and re.match(r'^\s*intent\s".*"\n$', line):
                attrs['intentLabels'] = re.split('"', line)[-1].strip()
            elif is_in_node and re.match(r'^\s*content\s".*"\n$', line):
                attrs['contentLabels'] = re.split('"', line)[-1].strip()
            elif is_in_node and re.match(r'^\s*likes\s\d+\n$', line):
                attrs['likes'] = re.findall(r'\d+', line.strip())[0]
            elif is_in_node and re.match(r'^\s*content\s".*"\n$', line):
                attrs['text'] = re.split(r'"', line)[-1].strip()
            elif is_in_edge and re.match(r'^\s*source\s\d+\n$', line):
                from_node = re.findall(r'\d+', line.strip())[0]
            elif is_in_edge and re.match(r'^\s*target\s\d+\n$', line):
                to_node = re.findall(r'\d+', line.strip())[0]
    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir',
                        help='Path to the directory containing .gml files' +
                             '\n For example, /home/user/data',
                        type=str,
                        default='.')

    argv = parser.parse_args()

    gml_paths = []
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(argv.data_dir):
        path = root.split(os.sep)

        print(root)
        print((len(path) - 1) * '---', os.path.basename(root))

        for file_name in files:
            if not file_name.endswith('.gml'):
                continue
            print(len(path) * '---', file_name)

            gml_paths.append(os.path.abspath(os.path.join(root, file_name)))

    for graph_path in gml_paths:
        g = load_graph(graph_path)
        print(dump_graph_for_graphviz(g))
        g = nx.read_gml(graph_path)
        print(g)
