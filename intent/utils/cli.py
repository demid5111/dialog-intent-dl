import argparse


def get_cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir',
                        help='Path to the directory containing .gml files' +
                             '\n For example, /home/user/data',
                        type=str,
                        default='.',
                        required=True)

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
    return parser
