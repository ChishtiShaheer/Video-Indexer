"""Pure-Python unit tests (no GPU, numpy or video needed).

Run:  python -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidindex import common as c  # noqa: E402


def counts(person=0, car=0, dog=0, laptop=0, phone=0):
    return (person, car, dog, laptop, phone)


class MedianTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(c.median_floor([]), 0)

    def test_odd(self):
        self.assertEqual(c.median_floor([5, 1, 3]), 3)

    def test_even_rounds_down(self):
        self.assertEqual(c.median_floor([1, 2]), 1)      # 1.5 -> 1
        self.assertEqual(c.median_floor([2, 3]), 2)      # 2.5 -> 2
        self.assertEqual(c.median_floor([0, 0, 1, 1]), 0)  # 0.5 -> 0
        self.assertEqual(c.median_floor([2, 2, 2, 4]), 2)  # 2.0 -> 2
        self.assertEqual(c.median_floor([1, 3]), 2)      # exactly 2


class SecondResultTests(unittest.TestCase):
    def test_no_frames(self):
        self.assertEqual(c.second_result([]), (0, 0))

    def test_present_when_at_least_half(self):
        # 2 of 4 frames contain a car -> present (>= half)
        s = [counts(car=1), counts(car=2), counts(), counts()]
        mask, people = c.second_result(s)
        self.assertTrue(mask & (1 << 1))
        # 1 of 4 -> not present
        s = [counts(car=1), counts(), counts(), counts()]
        mask, _ = c.second_result(s)
        self.assertFalse(mask & (1 << 1))

    def test_three_samples_need_two(self):
        s = [counts(dog=1), counts(), counts()]
        self.assertFalse(c.second_result(s)[0] & (1 << 2))
        s = [counts(dog=1), counts(dog=1), counts()]
        self.assertTrue(c.second_result(s)[0] & (1 << 2))

    def test_people_median(self):
        s = [counts(person=2), counts(person=3), counts(person=2)]
        self.assertEqual(c.second_result(s)[1], 2)
        s = [counts(person=1), counts(person=2)]
        self.assertEqual(c.second_result(s)[1], 1)  # 1.5 -> 1

    def test_mask_bits_follow_object_order(self):
        s = [counts(person=1, phone=1)]
        mask, _ = c.second_result(s)
        self.assertEqual(mask, 0b10001)


class FinalizeTests(unittest.TestCase):
    def parse(self, lines):
        return [json.loads(x) for x in lines]

    def test_all_seconds_written_including_empty(self):
        secs = {0: [counts(person=2, laptop=1)], 2: [counts()]}
        out = self.parse(c.finalize_video("f.parquet:3", secs, 3))
        self.assertEqual([o["second"] for o in out], [0, 1, 2, 3])
        self.assertEqual(out[0]["objects"], ["person", "laptop"])
        self.assertEqual(out[0]["people"], 2)
        self.assertEqual(out[1], {"video": "f.parquet:3", "second": 1,
                                  "objects": [], "people": 0})

    def test_error_keeps_earlier_seconds_and_adds_error_line(self):
        secs = {0: [counts(person=1)], 1: [counts(person=1)], 2: [counts(person=1)]}
        out = self.parse(c.finalize_video("v:0", secs, 2, error="boom"))
        self.assertEqual([("error" in o) for o in out], [False, False, True])


if __name__ == "__main__":
    unittest.main()
