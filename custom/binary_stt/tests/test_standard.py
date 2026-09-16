import unittest
import torch

from binary_stt.standard_model import StandardCTC
from binary_stt.standard_experiment import LogMel, prepare_row, TOKENIZER
from binary_stt.train import _collate, _loss


class StandardTests(unittest.TestCase):
    def test_real_feature_ctc_backward_and_reload(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        rows = [prepare_row({'audio': torch.randn(n) * .1, 'text': text})
                for n, text in [(16000, 'GO!'), (22400, 'Go home.')]]
        model = StandardCTC(width=32, layers=1, dropout=0).eval()
        x, lengths, targets, target_lengths = _collate(rows, LogMel(), torch.device('cpu'))
        logits, output_lengths = model(x, lengths)
        self.assertEqual(logits.shape[-1], TOKENIZER.vocab_size)
        self.assertEqual(output_lengths.tolist(), ((lengths + 1) // 2).tolist())
        loss = _loss(logits, output_lengths, targets, target_lengths, rows).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.subsample.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        copy = StandardCTC(width=32, layers=1, dropout=0).eval()
        copy.load_state_dict(model.state_dict())
        torch.testing.assert_close(copy(x, lengths)[0], logits)
        # Values outside true feature lengths cannot affect the prediction.
        altered = x.clone()
        altered[0, :, lengths[0]:] = 999
        torch.testing.assert_close(model(altered, lengths)[0], logits)

    def test_input_changes_logits(self):
        torch.manual_seed(456)
        model = StandardCTC(width=32, layers=1, dropout=0).eval()
        x = torch.randn(1, 80, 100)
        lengths = torch.tensor([100])
        self.assertGreater(float((model(x, lengths)[0] - model(x * 0, lengths)[0]).abs().mean()), .01)


if __name__ == '__main__':
    unittest.main()
