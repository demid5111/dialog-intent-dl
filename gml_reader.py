import networkx as nx
import re
import copy

from gml_utils import new_attrs


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
        content = graph_file.readlines()
        for idx, line in enumerate(content):
            if re.match('^\s*node\n$', line):
                is_in_node = True
                continue
            elif re.match('^\s*edge\n$', line):
                is_in_edge = True
                continue
            elif re.match('^\s*]\n$', line) and is_in_node:
                new_node_attrs = copy.deepcopy(attrs)
                new_id = new_node_attrs['commentID'] if new_node_attrs['commentID'] else new_node_attrs['postID']
                g.add_node(new_id, **new_node_attrs)
                attrs = new_attrs()
                is_in_node = False
                continue
            if re.match('^\s*]\n$', line) and is_in_edge:
                g.add_edge(from_node, to_node)
                from_node = None
                to_node = None
                is_in_edge = False
                continue

            if is_in_node and re.match('^\s*id\s\d+\n$', line):
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
