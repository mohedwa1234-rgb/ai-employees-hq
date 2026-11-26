#!/usr/bin/env python3
"""Run one agent request with sequential provider fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "company-config.json"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def provider_settings(provider: dict) -> tuple[str, str, str] | None:
    if provider.get("base_url_env"):
        base_url = os.environ.get(provider["base_url_env"], "")
    else:
        base_url = provider.get("base_url", "")

    api_key = os.environ.get(provider.get("key_env", ""), "")
    model = os.environ.get(provider.get("model_env", ""), "") or provider.get("default_model", "")

    if provider["name"] == "github_models" and not api_key:
        api_key = os.environ.get("GH_TOKEN", "")

    if not base_url or not api_key or not model:
        return None

    return base_url.rstrip("/"), api_key, model


def request_completion(base_url, api_key, model, messages, timeout, max_tokens):
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))

    text = result["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty provider response")
    return text.strip()


def run(args):
    config = load_config()["model_fallback"]
    agent_path = (ROOT / args.agent).resolve()
    if ROOT not in agent_path.parents or not agent_path.is_file():
        raise ValueError("Agent file must exist inside the repository.")

    agent = agent_path.read_text(encoding="utf-8")
    memory = (ROOT / "memory/shared-memory.md").read_text(encoding="utf-8")
    style = (ROOT / "memory/company-style.md").read_text(encoding="utf-8")

    messages = [
        {
            "role": "system",
            "content": (
                "أنت وكيل ضمن وكالة محتوى. اتبع تعليمات الدور والذاكرة وأسلوب الشركة. "
                "لا تكشف الأسرار أو مفاتيح API. لا تنفذ نشرًا أو إجراءً خارجيًا. "
                "المخرجات مسودات تحتاج موافقة بشرية.\n\n"
                f"تعليمات الدور:\n{agent}\n\n"
                f"أسلوب الشركة:\n{style}\n\n"
                f"الذاكرة المشتركة:\n{memory}"
            ),
        },
        {"role": "user", "content": args.task},
    ]

    failures = []
    for provider in sorted(config["providers"], key=lambda item: item["priority"]):
        settings = provider_settings(provider)
        if settings is None:
            failures.append(f"{provider['name']}: not configured")
            continue

        base_url, api_key, model = settings
        try:
            text = request_completion(
                base_url,
                api_key,
                model,
                messages,
                int(config["timeout_seconds"]),
                int(config["max_tokens"]),
            )
            generated = datetime.now(timezone.utc).isoformat()
            return (
                f"<!-- provider: {provider['name']} | model: {model} "
                f"| generated: {generated} -->\n\n{text}\n"
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            failures.append(f"{provider['name']}: {type(error).__name__}")

    raise RuntimeError(
        "All providers failed or were unavailable. " + "; ".join(failures)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = (ROOT / args.output).resolve()
    if ROOT not in output_path.parents:
        parser.error("Output path must remain inside the repository.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(run(args), encoding="utf-8")
        print(f"Generated {output_path.relative_to(ROOT)}")
        return 0
    except Exception as error:
        print(f"Agent run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())