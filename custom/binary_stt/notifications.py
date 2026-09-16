"""Durable local alerts, optional desktop alerts, and opt-in webhook delivery."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import smtplib
import ssl
import subprocess
import sys
from email.message import EmailMessage
from typing import Any, Mapping
from urllib import request
from urllib.parse import urlsplit, urlunsplit

from .storage import ensure_artifact_path


def validate_notification_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Check delivery prerequisites without contacting an external service.

    Set ``email_required`` to enforce this preflight; specifying ``email_to``
    makes it required by default. Secrets are read only from the environment.
    """
    config = dict(config or {})
    enabled = bool(config.get("email_enabled", True))
    recipient = config.get("email_to") if enabled else None
    required = enabled and bool(config.get("email_required", bool(recipient)))
    if recipient and ("\n" in str(recipient) or "\r" in str(recipient)):
        raise ValueError("Email recipient must be a single address without line breaks")
    host = os.environ.get("BINARY_STT_SMTP_HOST")
    sender = os.environ.get("BINARY_STT_SMTP_FROM")
    username = os.environ.get("BINARY_STT_SMTP_USER")
    password = os.environ.get("BINARY_STT_SMTP_PASSWORD")
    issues = []
    method = None
    if recipient:
        if host:
            if not sender:
                issues.append("BINARY_STT_SMTP_FROM is required")
            elif "\n" in sender or "\r" in sender:
                issues.append("BINARY_STT_SMTP_FROM contains a line break")
            if bool(username) != bool(password):
                issues.append("BINARY_STT_SMTP_USER and BINARY_STT_SMTP_PASSWORD must be set together")
            try:
                port = int(os.environ.get("BINARY_STT_SMTP_PORT", "587"))
                if not 1 <= port <= 65535:
                    raise ValueError
            except ValueError:
                issues.append("BINARY_STT_SMTP_PORT must be a valid port number")
            if not issues:
                method = "smtp"
        elif config.get("sendmail", False) and shutil.which("sendmail"):
            method = "sendmail"
        else:
            issues.append("Set BINARY_STT_SMTP_HOST and BINARY_STT_SMTP_FROM, or configure sendmail")
    elif required:
        issues.append("email_to must be configured for required email alerts")
    if required and issues:
        raise ValueError("Email alert preflight failed: " + "; ".join(issues))
    return {"email_enabled": enabled, "email_ready": bool(recipient and method), "email_method": method,
            "desktop_enabled": bool(config.get("desktop", True)), "issues": issues}


def _redact_text(value: str) -> str:
    for name in ("BINARY_STT_SMTP_PASSWORD", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                 "HUGGINGFACE_HUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, "[redacted]")

    def redact_url(match: re.Match) -> str:
        original = match.group(0)
        try:
            parsed = urlsplit(original)
            authority = parsed.netloc.rsplit("@", 1)[-1]
            return urlunsplit((parsed.scheme, authority, parsed.path,
                               "[redacted]" if parsed.query else "", ""))
        except ValueError:
            return "[redacted URL]"

    value = re.sub(r"https?://[^\s<>\"']+", redact_url, value)
    value = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)((?:password|api[_-]?key|access[_-]?token|HF_TOKEN)\s*[:=]\s*)[^\s,;]+",
                   r"\1[redacted]", value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        secret_keys = {"password", "token", "access_token", "api_key", "authorization", "secret"}
        return {str(k): "[redacted]" if str(k).lower() in secret_keys else _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _redact_text(str(value))


class Notifier:
    def __init__(self, run_dir: str | Path, config: Mapping[str, Any] | None = None):
        self.run_dir = Path(ensure_artifact_path(run_dir))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config or {})

    def check_delivery_config(self) -> dict[str, Any]:
        return validate_notification_config(self.config)

    def test(self) -> dict[str, Any]:
        self.check_delivery_config()
        return self.notify("notification_test", "Binary STT training alerts are configured.")

    def _email(self, event: str, detail: str, record: Mapping[str, Any]) -> None:
        config = validate_notification_config(self.config)
        if not config["email_ready"]:
            if self.config.get("email_to") and self.config.get("email_enabled", True):
                raise ValueError("Email delivery unavailable")
            return
        message = EmailMessage()
        message["To"] = str(self.config["email_to"])
        message["From"] = os.environ.get("BINARY_STT_SMTP_FROM", "binary-stt@localhost")
        message["Subject"] = f"[Binary STT] {event}"
        message.set_content(detail + "\n\n" + json.dumps(record, indent=2))
        timeout = min(max(float(self.config.get("email_timeout_seconds", 10)), 0.1), 10)
        if config["email_method"] == "sendmail":
            subprocess.run([shutil.which("sendmail"), "-t", "-oi"], input=message.as_bytes(), check=True,
                           timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        host = os.environ["BINARY_STT_SMTP_HOST"]
        port = int(os.environ.get("BINARY_STT_SMTP_PORT", "587"))
        context = ssl.create_default_context()
        if port == 465:
            connection = smtplib.SMTP_SSL(host, port, timeout=timeout, context=context)
        else:
            connection = smtplib.SMTP(host, port, timeout=timeout)
        with connection as smtp:
            if port != 465:
                smtp.ehlo()
                smtp.starttls(context=context)
                smtp.ehlo()
            if os.environ.get("BINARY_STT_SMTP_USER"):
                smtp.login(os.environ["BINARY_STT_SMTP_USER"], os.environ["BINARY_STT_SMTP_PASSWORD"])
            smtp.send_message(message)

    def notify(self, event: str, detail: str, **metrics: Any) -> dict[str, Any]:
        """Never raise on delivery failure or include a webhook endpoint in logs.

        Configure ``webhook_url_env`` with an environment variable name. The
        generic JSON endpoint is contacted only when explicitly configured.
        """
        record = _json_safe({
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event, "detail": detail,
            "run_dir": str(self.run_dir), **metrics,
        })
        event, detail = record["event"], record["detail"]
        payload = json.dumps(record, allow_nan=False, sort_keys=True).encode()
        failures: list[str] = []
        try:
            alert_path = ensure_artifact_path(self.run_dir / "alerts.jsonl")
            descriptor = os.open(alert_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                stream.write(payload + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            failures.append(f"local alert: {type(exc).__name__}")
        print(f"[binary-stt {event}] {detail}", file=sys.stderr, flush=True)
        try:
            self._email(event, detail, record)
        except Exception as exc:
            failures.append(f"email alert: {type(exc).__name__}")
        if self.config.get("desktop", True):
            try:
                subprocess.run(
                    ["notify-send", "--urgency=critical", "Binary STT training", f"{event}: {detail}"],
                    check=True, timeout=min(float(self.config.get("desktop_timeout_seconds", 2)), 5),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                if shutil.which("busctl"):
                    try:
                        subprocess.run([
                            "busctl", "--user", "call", "org.freedesktop.Notifications", "/org/freedesktop/Notifications",
                            "org.freedesktop.Notifications", "Notify", "susssasa{sv}i", "Binary STT", "0", "",
                            "Binary STT training", f"{event}: {detail}", "0", "0", "10000",
                        ], check=True, timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except (OSError, subprocess.SubprocessError) as dbus_exc:
                        failures.append(f"desktop alert: {type(dbus_exc).__name__}")
                else:
                    failures.append(f"desktop alert: {type(exc).__name__}")
        env_name = self.config.get("webhook_url_env")
        if env_name:
            endpoint = os.environ.get(str(env_name))
            if not endpoint:
                failures.append("webhook alert: configured environment variable is unset")
            elif not endpoint.startswith("https://"):
                failures.append("webhook alert: HTTPS endpoint required")
            else:
                try:
                    req = request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                    timeout = min(max(float(self.config.get("webhook_timeout_seconds", 3)), 0.1), 10)
                    with request.urlopen(req, timeout=timeout) as response:
                        if response.status >= 400:
                            failures.append("webhook alert: unsuccessful HTTP response")
                except Exception as exc:
                    # HTTPError and URLError can contain URLs and auth tokens.
                    failures.append(f"webhook alert: {type(exc).__name__}")
        for failure in failures:
            print(f"[binary-stt notification] {failure}", file=sys.stderr, flush=True)
        if failures:
            try:
                alert_path = ensure_artifact_path(self.run_dir / "alerts.jsonl")
                with alert_path.open("a") as stream:
                    stream.write(json.dumps({"time": record["time"], "event": "notification_delivery_failed",
                                             "original_event": event, "failures": failures}) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                pass
        return {"record": record, "delivery_failures": failures}
