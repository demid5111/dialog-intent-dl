import unittest

from intent.gml.gml_reader import load_graph
from intent.gml.gml_slicer import slice_graph_nx, slice_graph_recursive


class GMLStripperTest(unittest.TestCase):
    def test_strip_sample_2_paths(self):
        g = load_graph('./test/sample_0.gml')
        subgraphs = slice_graph_nx(g)
        self.assertEqual(len(subgraphs), 2)
        self.assertCountEqual(subgraphs, [
            ['0', '1', '3', '4'],
            ['0', '1', '2', '4'],
        ])

    def test_strip_sample_2_multiply_paths(self):
        g = load_graph('./test/sample_2.gml')
        subgraphs = slice_graph_nx(g)
        self.assertEqual(len(subgraphs), 4)
        self.assertCountEqual(subgraphs, [
            ['0', '1', '3', '4', '5'],
            ['0', '1', '3', '4', '6'],
            ['0', '1', '2', '4', '5'],
            ['0', '1', '2', '4', '6'],
        ])

    def test_strip_sample_4_paths(self):
        g = load_graph('./test/sample_1.gml')
        subgraphs = slice_graph_nx(g)
        self.assertEqual(len(subgraphs), 4)
        self.assertCountEqual(subgraphs, [
            ['0', '1', '3'],
            ['0', '1', '4', '6'],
            ['0', '1', '4', '7'],
            ['0', '2', '5'],
        ])


class GMLStripperComparedToNXTest(unittest.TestCase):
    def test_strip_sample_2_paths(self):
        g = load_graph('./test/sample_0.gml')
        subgraphs_nx = slice_graph_nx(g)
        subgraphs_rec = slice_graph_recursive(g)
        self.assertEqual(len(subgraphs_nx), len(subgraphs_rec))
        subgraphs_rec = [list(a) for a in subgraphs_rec]
        self.assertCountEqual(subgraphs_nx, subgraphs_rec)

    def test_strip_sample_2_multiply_paths(self):
        g = load_graph('./test/sample_2.gml')
        subgraphs_nx = slice_graph_nx(g)
        subgraphs_rec = slice_graph_recursive(g)
        self.assertEqual(len(subgraphs_nx), len(subgraphs_rec))
        subgraphs_rec = [list(a) for a in subgraphs_rec]
        self.assertCountEqual(subgraphs_nx, subgraphs_rec)

    def test_strip_sample_4_paths(self):
        g = load_graph('./test/sample_1.gml')
        subgraphs_nx = slice_graph_nx(g)
        subgraphs_rec = slice_graph_recursive(g)
        self.assertEqual(len(subgraphs_nx), len(subgraphs_rec))
        subgraphs_rec = [list(a) for a in subgraphs_rec]
        self.assertCountEqual(subgraphs_nx, subgraphs_rec)
