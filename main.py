import os

import gensim
import nltk
import time
import pandas as pd

from intent.gml.gml_distance_calculator import calculate_doc2vec, calculate_distance, calculate_cosine
from intent.post.post_reader import read_post_info, get_posts_col
from intent.utils.cli import get_cli_arguments
from intent.gml.gml_reader import load_graph
from intent.gml.gml_slicer import slice_graph_nx, clean_up_graph, remove_extra_intentions, embed_post_text
from intent.gml.gml_utils import dump_graph_csv
from intent.utils.io_utils import find_all_paths


def split_gml(graph_path, output_dir, vec_size, single_intent, ft_model, posts_index=None, distance='cosine'):
    file_name = graph_path.split(os.sep)[-1].split('.')[0]

    a1 = time.time()
    g = load_graph(graph_path)
    print('[INFO] Loading graph took {}s'.format(time.time() - a1))
    print('[INFO] Graph contains {} nodes and {} edges'.format(g.number_of_nodes(), g.number_of_edges()))

    if posts_index is not None:
        embed_post_text(g, posts_index)
    clean_up_graph(g)

    if single_intent:
        remove_extra_intentions(g)
        assert g.number_of_nodes() > 1, 'Graph can not be empty'
        print('[INFO] Reduced by keeping single intent.')
        print('[INFO] Graph contains {} nodes and {} edges'.format(g.number_of_nodes(), g.number_of_edges()))

    calculate_doc2vec(g, vec_size, ft_model)
    print('[INFO] Calculated doc2vec')

    subgraphs = slice_graph_nx(g)

    sub = [g.subgraph(s) for s in subgraphs]

    if distance != 'cosine':
        raise NotImplementedError('Only "cosine" distance metric is supported')
    for subgraph in sub:
        calculate_distance(subgraph, distance_lambda=calculate_cosine)
    print('[INFO] Calculated {} distance'.format(distance))

    file_name_template = os.path.join(output_dir, '{}_output_{{}}'.format(file_name))

    return g, sub, file_name_template


def split_gml_plain(graph_path, output_dir, single_intent, ft_model, vec_size, i, posts_index=None, distance='cosine'):
    _, sub, file_name_template = split_gml(graph_path, output_dir, vec_size, single_intent, ft_model,
                                           posts_index=posts_index, distance=distance)
    for idx, s in enumerate(sub):
        dump_graph_csv(s, file_name=file_name_template.format(idx))
    print('Finished with {}'.format(i))


def split_gml_light_concurrent(graph_path, output_dir, single_intent, ft_model, vec_size, i, posts_index=None,
                               distance='cosine'):
    g, subgraph_ids, file_name_template = split_gml(graph_path, output_dir, vec_size, single_intent, ft_model,
                                                    posts_index=posts_index, distance=distance)
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
    posts_paths = find_all_paths(argv.posts_dir, ext='.csv')

    try:
        os.makedirs(argv.output_dir)
    except FileExistsError:
        pass

    if argv.proxy:
        nltk.set_proxy(argv.proxy)

    nltk.download('stopwords')

    ft_model = gensim.models.fasttext.FastText.load(argv.model)

    frames = []
    for idx, post_path in enumerate(posts_paths):
        print('Reading posts: {}/{}, file: {}'.format(idx + 1, len(posts_paths), post_path))
        frames.append(read_post_info(post_path))

    # need to check for collisions
    print(frames[0])
    posts_index = pd.concat(frames, verify_integrity=True, ignore_index=True)

    for idx, graph_path in enumerate(gml_paths):
        print('Analyzing graphs: {}/{}, file: {}'.format(idx + 1, len(gml_paths), graph_path))
        if argv.mode == 'plain':
            split_gml_plain(graph_path, argv.output_dir, argv.single_intent, ft_model, vec_size=300, i=idx,
                            posts_index=posts_index, distance=argv.metric)
        else:
            split_gml_light_concurrent(graph_path, argv.output_dir, argv.single_intent, ft_model, vec_size=300, i=idx,
                                       posts_index=posts_index, distance=argv.metric)
