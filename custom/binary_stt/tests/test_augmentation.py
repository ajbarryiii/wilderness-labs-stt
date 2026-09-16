"""SpecAugment boundaries and checkpoint-compatible randomness."""

import unittest

import torch

from binary_stt.augmentation import spec_augment


class SpecAugmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_padding_and_inputs_unchanged(self):
        features = torch.ones(3, 80, 100)
        lengths = torch.tensor([100, 43, 0])
        features[1, :, 43:] = 7
        features[2] = 9
        original = features.clone()
        lengths_before = lengths.clone()
        result = spec_augment(features, lengths, {"enabled": True}, 0)
        torch.testing.assert_close(features, original)
        torch.testing.assert_close(lengths, lengths_before)
        torch.testing.assert_close(result[1, :, 43:], original[1, :, 43:])
        torch.testing.assert_close(result[2], original[2])
        self.assertTrue((result[0] == 0).any())
        self.assertNotEqual(result.data_ptr(), features.data_ptr())

    def test_torch_rng_restore_reproduces_masks(self):
        features = torch.ones(2, 80, 300)
        lengths = torch.tensor([300, 177])
        rng = torch.get_rng_state()
        first = spec_augment(features, lengths, {"enabled": True}, 4)
        torch.set_rng_state(rng)
        second = spec_augment(features, lengths, {"enabled": True}, 4)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_disabled_and_before_start_are_identity_without_rng_use(self):
        features = torch.randn(1, 80, 50)
        lengths = torch.tensor([50])
        for settings, step in ((None, 0), ({"enabled": False}, 10),
                               ({"enabled": True, "start_step": 10}, 9)):
            with self.subTest(settings=settings, step=step):
                rng = torch.get_rng_state()
                result = spec_augment(features, lengths, settings, step)
                torch.testing.assert_close(result, features, atol=0, rtol=0)
                torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                self.assertNotEqual(result.data_ptr(), features.data_ptr())
        active = spec_augment(features, lengths, {"enabled": True, "start_step": 10}, 10)
        self.assertFalse(torch.equal(active, features))

    def test_short_and_empty_examples_are_safe(self):
        features = torch.ones(4, 80, 19)
        lengths = torch.tensor([0, 1, 4, 19])
        result = spec_augment(features, lengths, {"enabled": True, "freq_masks": 0}, 0)
        # floor(0.05 * valid_frames) is zero throughout this batch.
        torch.testing.assert_close(result, features)
        single = spec_augment(torch.ones(1, 1, 1), torch.tensor([1]),
                              {"enabled": True, "max_time_fraction": 1.0}, 0)
        torch.testing.assert_close(single, torch.ones_like(single))

    def test_time_mask_widths_respect_each_example_fraction(self):
        features = torch.ones(2, 80, 400)
        lengths = torch.tensor([400, 100])
        settings = {"enabled": True, "freq_masks": 0, "time_masks": 2,
                    "max_time_width": 20, "max_time_fraction": 0.05}
        for _ in range(20):
            result = spec_augment(features, lengths, settings, 0)
            self.assertLessEqual(int((result[0, 0] == 0).sum()), 40)
            self.assertLessEqual(int((result[1, 0] == 0).sum()), 10)
            torch.testing.assert_close(result[1, :, 100:], features[1, :, 100:])
            for example in result:
                torch.testing.assert_close(example, example[0:1].expand_as(example))

    def test_invalid_configuration_and_lengths_fail(self):
        features = torch.ones(1, 80, 10)
        lengths = torch.tensor([10])
        for settings in ({"enabled": "yes"}, {"freq_width": -1},
                         {"time_masks": 1.5}, {"start_step": -1},
                         {"max_time_fraction": float("nan")},
                         {"max_time_fraction": 1.1}, {"time_mask": 2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                spec_augment(features, lengths, settings, 0)
        for bad_lengths in (torch.tensor([11]), torch.tensor([-1]),
                            torch.tensor([10.0]), torch.tensor([10, 9])):
            with self.subTest(lengths=bad_lengths), self.assertRaises(ValueError):
                spec_augment(features, bad_lengths, {"enabled": True}, 0)


if __name__ == "__main__":
    unittest.main()
