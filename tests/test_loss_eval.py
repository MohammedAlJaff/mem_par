import unittest

try:
    import torch
    from mp_config import clone_config, get_base_config
    from mp_metrics.loss_eval import collect_loss_eval_record
    from mp_tracking import loss_eval_metric_payload
except ModuleNotFoundError:
    torch = None
    clone_config = None
    get_base_config = None
    collect_loss_eval_record = None
    loss_eval_metric_payload = None


class TinyLossModel(torch.nn.Module if torch is not None else object):
    def compute_loss(self, x, eps=None, t=None):
        if eps is None:
            eps = torch.randn_like(x)
        if t is None:
            t = torch.rand(x.shape[0], device=x.device)
        per_sample = ((x + eps) ** 2).flatten(1).mean(1)
        return (per_sample * t).mean()


@unittest.skipIf(
    torch is None or get_base_config is None,
    "torch and config helpers are required",
)
class TestLossEval(unittest.TestCase):
    def test_loss_eval_record_is_deterministic_and_named(self):
        config = clone_config(get_base_config())
        config.training.seed = 11
        config.evaluation.batch_size = 2
        config.evaluation.loss_eval_batch_size = 2
        config.evaluation.loss_eval_times = (0.5, 0.75)

        model = TinyLossModel()
        model.train()
        eval_tensor = torch.ones(3, 4)

        first = collect_loss_eval_record(5, model, eval_tensor, config, "cpu")
        second = collect_loss_eval_record(5, model, eval_tensor, config, "cpu")
        payload = loss_eval_metric_payload(first)

        self.assertEqual(first, second)
        self.assertTrue(model.training)
        self.assertEqual(first["step"], 5)
        self.assertEqual(first["batch_size"], 2)
        self.assertIn("random_t_loss", first)
        self.assertIn("loss_t_0_5", first)
        self.assertIn("loss_t_0_75", first)
        self.assertIn("eval/random_t_loss", payload)
        self.assertIn("eval/loss_t_0_5", payload)
        self.assertIn("eval/loss_t_0_75", payload)


if __name__ == "__main__":
    unittest.main()
