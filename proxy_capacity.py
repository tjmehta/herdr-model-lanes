"""Read-only CLIProxyAPI capacity adapter. See THIRD_PARTY_NOTICES.md.

Only normalized capacity leaves this module. Never print HTTP errors, account
metadata, response bodies or credentials. The proxy remains the refresh owner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

URLS = {
    "claude": "https://api.anthropic.com/api/oauth/usage",
    "codex": "https://chatgpt.com/backend-api/wham/usage",
}
MAX_BYTES = 2 * 1024 * 1024


class CapacityError(Exception):
    """A fixed, credential-free state code."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CapacityError("unavailable")


def epoch(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise CapacityError("unknown")
    try:
        result = float(value)
    except (ValueError, TypeError):
        try:
            result = datetime.fromisoformat(value).timestamp()
        except (ValueError, TypeError, AttributeError):
            raise CapacityError("unknown") from None
    if not math.isfinite(result) or result <= 0:
        raise CapacityError("unknown")
    return result


def result(state, reset=None, remaining=None):
    return {"state": state, "reset_at": reset, "remaining_percent": remaining}


def window(value, used_key, now):
    if not isinstance(value, dict):
        raise CapacityError("unknown")
    used = value.get(used_key)
    if isinstance(used, bool) or not isinstance(used, (int, float)):
        raise CapacityError("unknown")
    if not math.isfinite(used) or used < 0:
        raise CapacityError("unknown")
    reset = epoch(value.get("resets_at", value.get("reset_at")))
    if reset is None and value.get("reset_after_seconds") is not None:
        seconds = value["reset_after_seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (float, int))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise CapacityError("unknown")
        reset = now + seconds
    return result(
        "stale" if reset is not None and reset <= now else "ok",
        reset,
        max(0, 100 - used),
    )


def combine(windows, now):
    """Known depletion wins over an unknown sibling window, until its reset."""
    depleted = [
        w
        for w in windows
        if w["state"] == "exhausted"
        or (w["state"] == "ok" and w["remaining_percent"] == 0)
    ]
    if depleted:
        resets = [w["reset_at"] for w in depleted]
        return result("exhausted", max(resets) if all(resets) else None, 0)
    for state in ("auth_error", "unavailable", "unknown", "stale"):
        if any(w["state"] == state for w in windows):
            return result(state)
    if not windows:
        return result("unknown")
    tightest = min(windows, key=lambda w: w["remaining_percent"])
    return dict(tightest)


def parse_usage(provider, body, model, now):
    """Apply shared AND relevant model windows; credits never enable a lane."""
    if not isinstance(body, dict):
        return result("unknown")
    windows = []

    def add(value, key):
        try:
            windows.append(window(value, key, now))
        except CapacityError as exc:
            windows.append(result(str(exc)))

    if provider == "claude":
        for required in ("five_hour", "seven_day"):
            add(body.get(required), "utilization")
        for name, value in body.items():
            if name in ("five_hour", "seven_day") or value is None:
                continue
            if not name.startswith(("seven_day_", "five_hour_")):
                continue
            # Known named pools can be scoped. Unrecognized pools conservatively
            # constrain every model, rather than inventing independent capacity.
            if (
                name in ("seven_day_sonnet", "seven_day_opus")
                and name.removeprefix("seven_day_") not in model.lower()
            ):
                continue
            add(value, "utilization")
    else:
        rates = [body.get("rate_limit")]
        additional = body.get("additional_rate_limits")
        if additional is None:
            additional = []
        if isinstance(additional, dict):
            if any(not isinstance(v, dict) for v in additional.values()):
                windows.append(result("unknown"))
            additional = [
                dict(v, limit_name=k)
                for k, v in additional.items()
                if isinstance(v, dict)
            ]
        if not isinstance(additional, list):
            windows.append(result("unknown"))
            additional = []
        for item in additional:
            if not isinstance(item, dict):
                windows.append(result("unknown"))
                continue
            name = str(item.get("limit_name", "")).lower()
            if name.startswith("gpt-") and name != model.lower():
                continue
            rates.append(item.get("rate_limit", item))
        for rate in rates:
            if not isinstance(rate, dict):
                windows.append(result("unknown"))
                continue
            parsed = []
            for key in ("primary_window", "secondary_window"):
                if rate.get(key) is not None:
                    before = len(windows)
                    add(rate[key], "used_percent")
                    parsed.extend(windows[before:])
            if rate.get("allowed") is False or rate.get("limit_reached") is True:
                resets = [w["reset_at"] for w in parsed if w["remaining_percent"] == 0]
                reset = max(resets) if resets and all(resets) else None
                windows.append(
                    result("stale" if reset and reset <= now else "exhausted", reset, 0)
                )
            elif not parsed:
                windows.append(result("unknown"))
        usage = body.get("model_usage")
        if usage is None:
            usage = {}
        if not isinstance(usage, dict):
            windows.append(result("unknown"))
            usage = {}
        if model in usage:
            entry = usage[model]
            if not isinstance(entry, dict) or not isinstance(
                entry.get("available"), bool
            ):
                windows.append(result("unknown"))
            elif not entry["available"]:
                try:
                    reset = epoch(entry.get("available_at"))
                    windows.append(
                        result(
                            "stale" if reset and reset <= now else "exhausted", reset, 0
                        )
                    )
                except CapacityError:
                    windows.append(result("unknown"))
    return combine(windows, now)


def restriction(account, model, now):
    if account.get("disabled") or account.get("status") == "disabled":
        return result("disabled")
    blocks = []
    for cooldown in account.get("cooldowns") or []:
        if cooldown.get("scope") != "credential" and cooldown.get("model_key") != model:
            continue
        reset = epoch(cooldown.get("retry_at"))
        if reset is None or reset <= now:
            continue
        reason = cooldown.get("reason")
        state = "unavailable"
        if reason in ("credential_quota", "quota"):
            state = "exhausted"
        elif reason in ("unauthorized", "invalid_grant", "payment_required"):
            state = "auth_error"
        blocks.append(result(state, reset, 0 if state == "exhausted" else None))
    if blocks:
        return combine(blocks, now)
    if account.get("unavailable") or account.get("status") == "error":
        # Legacy APIs don't expose the reason. Never guess quota from free text.
        return result("unavailable", epoch(account.get("next_retry_after")))
    return None


def read_key(config):
    name = config.get("key_env", "CLIPROXYAPI_MANAGEMENT_KEY")
    key = os.environ.get(name)
    if not key and config.get("key_file"):
        path = Path(config["key_file"]).expanduser()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o077
            ):
                raise CapacityError("insecure_key_file")
            value = stream.read(65537)
        if len(value) > 65536:
            raise CapacityError("invalid_key_file")
        # A raw key or a protected dotenv file. Do not execute/source the file.
        assignments = [
            line.split("=", 1)[1].strip().strip("\"'")
            for line in value.splitlines()
            if line.startswith(name + "=")
        ]
        key = assignments[0] if assignments else value.strip()
    if not key or "\n" in key or "\r" in key:
        raise CapacityError("missing_management_key")
    return key


class Client:
    def __init__(self, config):
        self.base = config["management_url"].rstrip("/")
        url = urllib.parse.urlsplit(self.base)
        if (
            url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in ("", "/v0/management")
        ):
            raise CapacityError("invalid_management_url")
        if url.scheme != "https" and not (
            url.scheme == "http" and url.hostname in ("127.0.0.1", "::1", "localhost")
        ):
            raise CapacityError("insecure_management_url")
        if not self.base.endswith("/v0/management"):
            self.base += "/v0/management"
        self.key = read_key(config)
        self.timeout = min(10, max(0.1, float(config.get("timeout_seconds", 4))))
        self.deadline = time.monotonic() + min(
            60, max(1, float(config.get("budget_seconds", 20)))
        )
        self.opener = urllib.request.build_opener(NoRedirect)

    def request(self, path, data=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CapacityError("unavailable")
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self.opener.open(
                request, timeout=min(self.timeout, remaining)
            ) as response:
                raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise CapacityError("unknown")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise CapacityError(
                "auth_error" if exc.code in (401, 403) else "unavailable"
            ) from None
        except (OSError, ValueError, urllib.error.URLError):
            raise CapacityError("unavailable") from None

    def usage(self, account, provider):
        headers = {"Authorization": "Bearer $TOKEN$", "Accept": "application/json"}
        if provider == "claude":
            headers["anthropic-beta"] = "oauth-2025-04-20"
        else:
            account_id = account.get("account_id") or (
                account.get("id_token") or {}
            ).get("chatgpt_account_id")
            if account_id:
                headers["ChatGPT-Account-Id"] = account_id
        data = self.request(
            "/api-call",
            {
                "auth_index": account["auth_index"],
                "method": "GET",
                "url": URLS[provider],
                "header": headers,
            },
        )
        status = data.get("status_code")
        if status in (401, 403):
            raise CapacityError("auth_error")
        if status != 200:
            # 429 on a quota endpoint is not evidence that inference is depleted.
            raise CapacityError("unavailable")
        body = data.get("body")
        try:
            return json.loads(body) if isinstance(body, str) else body
        except ValueError:
            raise CapacityError("unknown") from None


def collect(client, requested, now):
    """Return per-lane anonymous account states, never account identifiers."""
    accounts = client.request("/auth-files").get("files")
    if not isinstance(accounts, list) or len(accounts) > 64:
        raise CapacityError("unknown")
    output = {f"{p}/{m}": [] for p, m in requested}
    seen = set()
    for account in accounts:
        provider = account.get("provider")
        models = [m for p, m in requested if p == provider]
        if not models or account.get("disabled") or account.get("status") == "disabled":
            continue
        identity = account.get("auth_index")
        if not identity or (provider, identity) in seen:
            continue
        seen.add((provider, identity))
        try:
            listed = client.request(
                "/auth-files/models?"
                + urllib.parse.urlencode({"name": account["name"]})
            ).get("models")
            if not isinstance(listed, list):
                raise CapacityError("unknown")
            supported = {item["id"] for item in listed}
        except CapacityError as exc:
            for model in models:
                output[f"{provider}/{model}"].append(result(str(exc)))
            continue
        eligible = [m for m in models if m in supported]
        pending = []
        for model in eligible:
            block = restriction(account, model, now)
            if block:
                output[f"{provider}/{model}"].append(block)
            else:
                pending.append(model)
        if pending:
            try:
                body = client.usage(account, provider)
                for model in pending:
                    output[f"{provider}/{model}"].append(
                        parse_usage(provider, body, model, now)
                    )
            except CapacityError as exc:
                for model in pending:
                    output[f"{provider}/{model}"].append(result(str(exc)))
    return output


def snapshot(config, requested, cache_dir, force=False, now=None):
    now = time.time() if now is None else now
    requested = sorted(set(requested))
    # Cache only anonymous, normalized decisions. Include source and model set.
    identity = json.dumps([1, config, requested], sort_keys=True).encode()
    path = cache_dir / ("proxy-" + hashlib.sha256(identity).hexdigest()[:24] + ".json")
    ttl = min(60, max(0, float(config.get("cache_seconds", 30))))
    if not force:
        try:
            cached = json.loads(path.read_text())
            age = now - cached["fetched_at"]
            resets = [
                v["reset_at"]
                for values in cached["capacity"].values()
                for v in values
                if v["reset_at"]
            ]
            if 0 <= age < ttl and all(reset > now for reset in resets):
                return cached
        except (OSError, ValueError, KeyError, TypeError):
            pass
    client = Client(config)
    try:
        capacity = collect(client, requested, now)
    except CapacityError as exc:
        capacity = {f"{p}/{m}": [result(str(exc))] for p, m in requested}
    data = {"fetched_at": now, "capacity": capacity}
    # Failed reads never resurrect an older healthy snapshot.
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    import tempfile

    fd, temporary = tempfile.mkstemp(dir=cache_dir, prefix=".proxy-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return data
