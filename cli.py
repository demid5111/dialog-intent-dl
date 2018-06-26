import argparse


def get_cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir',
                        help='Path to the directory containing .gml files' +
                             '\n For example, /home/user/data',
                        type=str,
                        default='.')

    parser.add_argument('--output-dir',
                        help='Path to the directory containing output .csv files' +
                             '\n For example, /home/user/output',
                        type=str,
                        default='.')
    return parser
