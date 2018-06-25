import argparse

from gml_reader import load_graph
from gml_slicer import slice_graph
from gml_utils import dump_graph_for_graphviz
from io_utils import find_all_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir',
                        help='Path to the directory containing .gml files' +
                             '\n For example, /home/user/data',
                        type=str,
                        default='.')

    argv = parser.parse_args()

    gml_paths = find_all_paths(argv.data_dir, ext='.gml')

    for graph_path in gml_paths:
        g = load_graph(graph_path)
        print(dump_graph_for_graphviz(g))

        subgraphs = slice_graph(g)

        sub = [g.subgraph(s) for s in subgraphs]

        print(sub)
