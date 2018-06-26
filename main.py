import os

from cli import get_cli_arguments
from gml_reader import load_graph
from gml_slicer import slice_graph
from gml_utils import dump_graph_csv
from io_utils import find_all_paths


def split_gml(graph_path, argv, i):
    file_name = graph_path.split(os.sep)[-1].split('.')[0]
    g = load_graph(graph_path)

    subgraphs = slice_graph(g)

    sub = [g.subgraph(s) for s in subgraphs]
    file_name_template = os.path.join(argv.output_dir, '{}_output_{{}}'.format(file_name))
    for idx, s in enumerate(sub):
        dump_graph_csv(s, file_name=file_name_template.format(idx))
    print('Finished with {}'.format(i))


def split_gml_light_concurrent(graph_path, argv, i):
    file_name = graph_path.split(os.sep)[-1].split('.')[0]
    g = load_graph(graph_path)

    subgraph_ids = slice_graph(g)

    file_name_template = os.path.join(argv.output_dir, '{}_output_{{}}'.format(file_name))

    import time
    from gevent.pool import Pool

    NUM_WORKERS = 4

    start_time = time.time()

    pool = Pool(NUM_WORKERS)
    for idx, s in enumerate(subgraph_ids):
        pool.spawn(dump_graph_csv, g, file_name_template.format(idx), s)

    pool.join()

    end_time = time.time()

    print("Time for pool: %ssecs" % (end_time - start_time))
    print('Finished with {}'.format(i))


if __name__ == '__main__':
    argv = get_cli_arguments().parse_args()

    gml_paths = find_all_paths(argv.data_dir, ext='.gml')

    for idx, graph_path in enumerate(gml_paths):
        print('Analyzing {}/{} {}'.format(idx, len(gml_paths)-1, graph_path))
        split_gml_light_concurrent(graph_path, argv, idx)
