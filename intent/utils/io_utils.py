import os


def find_all_paths(data_dir, ext=''):
    gml_paths = []
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(data_dir):
        path = root.split(os.sep)

        print(root)
        print((len(path) - 1) * '---', os.path.basename(root))

        for file_name in files:
            if ext and not file_name.endswith(ext):
                continue
            print(len(path) * '---', file_name)

            gml_paths.append(os.path.abspath(os.path.join(root, file_name)))
    return gml_paths
