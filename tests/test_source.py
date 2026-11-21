"""Pure-Python tests for dataset source traversal and file finding."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidindex import source as s  # noqa: E402


class SourceTests(unittest.TestCase):
    def test_find_parquet_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "train-00001.parquet")
            f2 = os.path.join(tmpdir, "train-00000.parquet")
            f3 = os.path.join(tmpdir, "other.txt")
            open(f1, "w").close()
            open(f2, "w").close()
            open(f3, "w").close()

            found = s.find_parquet_files(tmpdir)
            self.assertEqual(len(found), 2)
            self.assertEqual(os.path.basename(found[0]), "train-00000.parquet")
            self.assertEqual(os.path.basename(found[1]), "train-00001.parquet")

    def test_find_single_file(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            found = s.find_parquet_files(tmp.name)
            self.assertEqual(found, [tmp.name])


if __name__ == "__main__":
    unittest.main()
