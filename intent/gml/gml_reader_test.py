import unittest

from intent.gml.gml_reader import load_graph, slice_graph


class GMLStripperTest(unittest.TestCase):
    def test_strip_sample_2_paths(self):
        g = load_graph('./data/sample_0.gml')
        subgraphs = slice_graph(g)
        self.assertEqual(len(subgraphs), 2)
        self.assertCountEqual(subgraphs, [
            ('0', '1', '3', '4'),
            ('0', '1', '2', '4'),
        ])

    def test_strip_sample_4_paths(self):
        g = load_graph('./data/sample_1.gml')
        subgraphs = slice_graph(g)
        self.assertEqual(len(subgraphs), 4)
        self.assertCountEqual(subgraphs, [
            ('0', '1', '3'),
            ('0', '1', '4', '6'),
            ('0', '1', '4', '7'),
            ('0', '2', '5'),
        ])
