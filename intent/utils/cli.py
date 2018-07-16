import argparse


def get_cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir',
                        help='Path to the directory containing .gml files' +
                             '\n For example, /home/user/data',
                        type=str,
                        default='.',
                        required=True)
    parser.add_argument('--posts-dir',
                        help='Path to the directory containing .csv files with information about' +
                             ' posts.' +
                             '\n For example, /home/user/posts',
                        type=str,
                        default='.',
                        required=True)

    parser.add_argument('--metric',
                        help='Distance metric for comments and posts.' +
                             '\n For example, "cosine"',
                        choices=['cosine',],
                        default='cosine')

    parser.add_argument('--model',
                        help='The path to the model. Before running the tool, ' +
                             'proceed with README instructions to download it.',
                        type=str,
                        default='',
                        required=True)

    parser.add_argument('--output-dir',
                        help='Path to the directory containing output .csv files' +
                             '\n For example, /home/user/output',
                        type=str,
                        default='.')

    parser.add_argument('--proxy',
                        help='The tool needs to download the models. If you are' +
                             'behind the corporate proxy use this option to pass its value'
                             '\n For example, ' +
                             '--proxy http://user:password@corp-proxy:1024',
                        type=str,
                        default='')

    parser.add_argument('--mode',
                        help='The mode to run the tool. Either "plain"(default) or ' +
                             '"concurrent" to parallel routine of traversing the graph and ' +
                             'creating the file per each sequence',
                        choices=['plain', 'concurrent'],
                        default='plain')

    parser.add_argument('--single-intent',
                        help='The graph can contain comments with numerous intentions. ' +
                             'Using this option reduces number of paths due to ignorance ' +
                             'of all intentions except the main one',
                        action='store_true')

    parser.add_argument('--only-distance',
                        help='Include in the output file only distances without vectors for comments',
                        action='store_true')

    parser.add_argument('--output-format',
                        help='Format of the output file: csv or hdf5. For hdf5 format, the key is "data"',
                        choices=['csv', 'hdf5'],
                        default='csv')
    return parser
