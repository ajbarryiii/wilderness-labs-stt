"""CPU-only failure injection. No desktop, webhook, or email is actually sent."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from binary_stt.health import HealthMonitor, TrainingCollapse
from binary_stt.notifications import Notifier, validate_notification_config
from binary_stt.storage import ROOT, ensure_artifact_path
from binary_stt.supervisor import run_supervised


class HealthTests(unittest.TestCase):
    def monitor(self, **config):
        return HealthMonitor({"loss_window": 3, "min_train_steps": 3,
                              "loss_explosion_patience": 3, "zero_grad_patience": 3,
                              "phase_grace_steps": 2, "val_patience": 2, **config})

    def test_nonfinite_loss_and_gradient_always_stop(self):
        for loss, grad in [(float("nan"), 1), (1, float("inf"))]:
            monitor = self.monitor()
            monitor.observe_train(0, 1, 1, quantization=0)
            monitor.observe_train(1, 1, 1, quantization=0.1)
            with self.assertRaises(TrainingCollapse):
                monitor.observe_train(2, loss, grad, quantization=0.2)

    def test_loss_explosion_does_not_poison_baseline(self):
        monitor = self.monitor(loss_explosion_patience=10)
        for step in range(5):
            monitor.observe_train(step, 2, 1)
        for step in range(5, 14):
            monitor.observe_train(step, 200, 1)
        self.assertEqual(monitor.loss_reference, 2)
        with self.assertRaisesRegex(TrainingCollapse, "loss explosion"):
            monitor.observe_train(14, 200, 1)

    def test_isolated_loss_spike_recovers(self):
        monitor = self.monitor()
        for step in range(5):
            monitor.observe_train(step, 2, 1)
        monitor.observe_train(5, 200, 1)
        monitor.observe_train(6, 2, 1)
        self.assertEqual(monitor.loss_bad_count, 0)

    def test_continuously_changing_ramp_does_not_extend_grace(self):
        monitor = self.monitor()
        for step in range(5):
            monitor.observe_train(step, 2, 1, quantization=0)
        for step in range(5, 9):
            monitor.observe_train(step, 200, 1, quantization=(step - 4) / 10)
        with self.assertRaisesRegex(TrainingCollapse, "loss explosion"):
            monitor.observe_train(9, 200, 1, quantization=0.5)

    def test_sustained_zero_gradient_stops(self):
        monitor = self.monitor()
        for step in range(5):
            monitor.observe_train(step, 2, 1)
        monitor.observe_train(5, 2, 0)
        monitor.observe_train(6, 2, 0)
        with self.assertRaisesRegex(TrainingCollapse, "zero gradient"):
            monitor.observe_train(7, 2, 0)

    def test_weight_activation_schedule_has_separate_bounded_phases(self):
        monitor = self.monitor()
        monitor.observe_train(0, 1, 1, quantization={"weight": 0, "activation": 0})
        monitor.observe_train(1, 100, 1, quantization={"weight": 1, "activation": 0})
        self.assertEqual(monitor.grace_until_step, 3)
        monitor.observe_train(4, 100, 1, quantization={"weight": 1, "activation": 0.2})
        self.assertEqual(monitor.grace_until_step, 6)
        monitor.observe_train(5, 100, 1, quantization={"weight": 1, "activation": 0.4})
        self.assertEqual(monitor.grace_until_step, 6)

    def test_early_ctc_blanks_and_plateau_do_not_stop(self):
        monitor = self.monitor()
        for step in range(20):
            monitor.observe_validation(step, {"wer": 1.0, "empty_fraction": 1.0, "blank_fraction": 1.0})
        self.assertFalse(monitor.validation_armed)
        for step in range(20, 100):
            monitor.observe_validation(step, {"wer": 0.4, "empty_fraction": 0.0, "blank_fraction": 0.99})
        self.assertTrue(monitor.validation_armed)

    def test_validation_collapse_arms_after_learning(self):
        monitor = self.monitor()
        monitor.observe_validation(0, {"wer": 0.25, "empty_fraction": 0})
        monitor.observe_validation(1, {"wer": 1, "empty_fraction": 1, "blank_fraction": 1})
        with self.assertRaisesRegex(TrainingCollapse, "validation collapse"):
            monitor.observe_validation(2, {"wer": 1, "empty_fraction": 1, "blank_fraction": 1})

    def test_nonfinite_validation_stops_before_arming_and_during_grace(self):
        monitor = self.monitor()
        monitor.grace_until_step = 100
        with self.assertRaisesRegex(TrainingCollapse, "Nonfinite validation"):
            monitor.observe_validation(0, {"wer": float("nan")})

    def test_state_restores_pending_alarm_and_baseline(self):
        original = self.monitor()
        for step in range(5):
            original.observe_train(step, 2, 1)
        original.observe_train(5, 200, 1)
        restored = self.monitor()
        restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
        restored.observe_train(6, 200, 1)
        with self.assertRaises(TrainingCollapse):
            restored.observe_train(7, 200, 1)


class DiskTestCase(unittest.TestCase):
    def setUp(self):
        if not os.path.ismount("/mnt/hd"):
            self.skipTest("Artifact disk is not mounted")
        parent = ensure_artifact_path(ROOT / "tests")
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="health-", dir=parent))
        self.stderr = contextlib.redirect_stderr(io.StringIO())
        self.stderr.__enter__()

    def tearDown(self):
        self.stderr.__exit__(None, None, None)
        shutil.rmtree(self.directory)


class NotificationTests(DiskTestCase):
    def test_durable_alert_preserves_nonfinite_metrics(self):
        notifier = Notifier(self.directory, {"desktop": False, "email_enabled": False})
        result = notifier.notify("collapsed", "Nonfinite loss", loss=float("nan"))
        self.assertFalse(result["delivery_failures"])
        saved = json.loads((self.directory / "alerts.jsonl").read_text())
        self.assertEqual(saved["loss"], "nan")

    @patch.dict(os.environ, {"HF_TOKEN": "hf_secret_example"})
    def test_failure_alerts_redact_signed_urls_and_known_secrets(self):
        notifier = Notifier(self.directory, {"desktop": False, "email_enabled": False})
        notifier.notify("failed", "HTTP error https://user:password@example.com/data?X-Amz-Signature=abc hf_secret_example",
                        token="secret", exception="Authorization: Bearer abc123")
        saved = (self.directory / "alerts.jsonl").read_text()
        for forbidden in ("X-Amz-Signature", "user:password", "hf_secret_example", "abc123", '"token": "secret"'):
            self.assertNotIn(forbidden, saved)

    @patch.dict(os.environ, {}, clear=True)
    def test_required_email_refuses_unconfigured_launch(self):
        with self.assertRaisesRegex(ValueError, "Email alert preflight failed"):
            validate_notification_config({"email_to": "operator@example.com"})
        self.assertFalse(validate_notification_config({"email_to": "operator@example.com", "email_enabled": False})["email_ready"])

    @patch.dict(os.environ, {"BINARY_STT_SMTP_HOST": "example.invalid", "BINARY_STT_SMTP_FROM": "train@example.com"}, clear=True)
    @patch("binary_stt.notifications.smtplib.SMTP")
    def test_smtp_requires_starttls_before_delivery(self, smtp_class):
        smtp = smtp_class.return_value.__enter__.return_value
        result = Notifier(self.directory, {"desktop": False, "email_to": "operator@example.com"}).test()
        self.assertFalse(result["delivery_failures"])
        self.assertEqual([call[0] for call in smtp.method_calls], ["ehlo", "starttls", "ehlo", "send_message"])

    @patch.dict(os.environ, {"BINARY_STT_SMTP_HOST": "secret-host.invalid", "BINARY_STT_SMTP_FROM": "train@example.com"}, clear=True)
    @patch("binary_stt.notifications.smtplib.SMTP")
    def test_smtp_tls_failure_stops_delivery_and_redacts_error(self, smtp_class):
        smtp = smtp_class.return_value.__enter__.return_value
        smtp.starttls.side_effect = RuntimeError("secret-host.invalid?password=do-not-log")
        result = Notifier(self.directory, {"desktop": False, "email_to": "operator@example.com"}).notify("collapsed", "NaN")
        smtp.send_message.assert_not_called()
        self.assertEqual(result["delivery_failures"], ["email alert: RuntimeError"])
        self.assertNotIn("secret-host", (self.directory / "alerts.jsonl").read_text())
        self.assertEqual(len((self.directory / "alerts.jsonl").read_text().splitlines()), 2)

    @patch.dict(os.environ, {"BINARY_STT_SMTP_HOST": "example.invalid", "BINARY_STT_SMTP_PORT": "465",
                            "BINARY_STT_SMTP_FROM": "train@example.com"}, clear=True)
    @patch("binary_stt.notifications.smtplib.SMTP_SSL")
    def test_port_465_uses_tls_from_connection_start(self, smtp_class):
        smtp = smtp_class.return_value.__enter__.return_value
        Notifier(self.directory, {"desktop": False, "email_to": "operator@example.com"}).notify("test", "test")
        smtp.starttls.assert_not_called()
        smtp.send_message.assert_called_once()


class SupervisorTests(DiskTestCase):
    def supervise(self, code, **kwargs):
        return run_supervised([sys.executable, "-c", code], self.directory, timeout_seconds=0.3,
                              poll_interval=0.02, terminate_grace_seconds=0.05,
                              notification_config={"desktop": False, "email_enabled": False}, **kwargs)

    def test_hung_child_is_killed_without_touching_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            result = self.supervise("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)")
            self.assertEqual(result, 124)
            self.assertIsNone(unrelated.poll())
            status = json.loads((self.directory / "supervisor_status.json").read_text())
            self.assertEqual(status["reason"], "heartbeat_timeout")
            with self.assertRaises(ProcessLookupError):
                os.kill(status["pid"], 0)
        finally:
            unrelated.terminate()
            unrelated.wait()

    def test_sigkill_is_reported(self):
        self.assertEqual(self.supervise("import os,signal; os.kill(os.getpid(),signal.SIGKILL)"), 137)
        alert = json.loads((self.directory / "alerts.jsonl").read_text())
        self.assertEqual(alert["event"], "worker_failed")

    def test_clean_exit_requires_worker_terminal_status(self):
        self.assertEqual(self.supervise("pass"), 70)

    def test_preexisting_completed_status_cannot_hide_unexpected_exit(self):
        (self.directory / "status.json").write_text(json.dumps({"status": "completed"}))
        self.assertEqual(self.supervise("pass"), 70)

    def test_clean_terminal_status(self):
        target = repr(str(self.directory / "status.json"))
        self.assertEqual(self.supervise(f"import json; open({target},'w').write(json.dumps({{'status':'completed'}}))"), 0)

    def test_progress_heartbeat_extends_timeout(self):
        target = repr(str(self.directory))
        code = f"""import json,os,time
from pathlib import Path
p=Path({target})
for i in range(6):
    tmp=p/'heartbeat.tmp'
    tmp.write_text(json.dumps({{'time':time.time(),'stage':'training','pid':os.getpid()}}))
    tmp.replace(p/'heartbeat.json')
    time.sleep(.1)
(p/'status.json').write_text(json.dumps({{'status':'completed'}}))
"""
        self.assertEqual(self.supervise(code), 0)

    def test_stage_specific_deadline_allows_checkpoint(self):
        target = repr(str(self.directory))
        code = f"""import json,os,time
from pathlib import Path
p=Path({target})
(p/'heartbeat.json').write_text(json.dumps({{'time':time.time(),'stage':'checkpoint','pid':os.getpid()}}))
time.sleep(.5)
(p/'status.json').write_text(json.dumps({{'status':'completed'}}))
"""
        self.assertEqual(self.supervise(code, stage_timeouts={"checkpoint": 1}), 0)


if __name__ == "__main__":
    unittest.main()
