"""Stateless live completions, with a local response cache and a per-run call budget."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODELS = {
    "anthropic-http": "claude-sonnet-4-6",
    "claude-cli": "claude-sonnet-4-6",
    "openai-http": "gpt-5",
    "codex-cli": "gpt-5",
    "zai-anthropic": "glm-5.3",
}
ENDPOINTS = {
    "openai-http": "https://api.openai.com/v1/chat/completions",
    "anthropic-http": "https://api.anthropic.com/v1/messages",
    "zai-anthropic": "https://api.z.ai/api/anthropic/v1/messages",
}
MAX_TOKENS = 1024


class BackendError(RuntimeError):
    pass


class BudgetExhausted(BackendError):
    pass


@dataclass
class Completion:
    text: str
    usage: dict
    latency_s: float = 0
    cached: bool = False
    digest: str = ""


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        tmp = Path(out.name)
        try:
            json.dump(value, out, ensure_ascii=False)
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


@contextmanager
def cache_lock(cache: Path, digest: str):
    """Serialize identical misses across workers and restarted/concurrent processes."""
    directory = cache / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / digest).open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield  # Closing the file releases the OS lock, including after a crash.


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise BackendError("provider redirected the credentialed request")


def anthropic_usage(data):
    return {k: data[k] for k in (
        "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"
    ) if isinstance(data.get(k), int)}


@dataclass
class LLMBackend:
    name: str
    model: str
    cache: Path = Path("runs/cache")
    max_calls: int = 40
    cli: str | None = None
    timeout: int = 90
    calls: int = field(init=False, default=0)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def complete(self, system: str, prompt: str) -> Completion:
        identity = [2, self.name, self.model, self.cli, MAX_TOKENS, system, prompt]
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        path = self.cache / (digest + ".json")
        if path.exists():
            return Completion(**{**json.loads(path.read_text()), "cached": True})
        with cache_lock(self.cache, digest):
            if path.exists():
                return Completion(**{**json.loads(path.read_text()), "cached": True})
            with self._lock:
                if self.calls >= self.max_calls:
                    raise BudgetExhausted("live call budget exhausted")
                self.calls += 1  # Attempts, including transport failures.
            start = time.monotonic()
            try:
                result = self._request(system, prompt)
            except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.TimeoutExpired) as exc:
                raise BackendError(f"{self.name} request failed ({type(exc).__name__})") from None
            if not isinstance(result.text, str) or not result.text.strip():
                raise BackendError(f"{self.name} returned no text")
            result.latency_s = round(time.monotonic() - start, 4)
            result.digest = digest
            atomic_json(path, asdict(result))
            return result

    def _request(self, system, prompt):
        if self.name in ("claude-cli", "codex-cli"):
            return self._cli(system, prompt)
        openai = self.name == "openai-http"
        env_key = {"openai-http": "OPENAI_API_KEY", "anthropic-http": "ANTHROPIC_API_KEY",
                   "zai-anthropic": "ZAI_API_KEY"}[self.name]
        key = os.environ.get(env_key)
        if not key and self.name == "zai-anthropic":
            key = (Path.home() / ".zai-key").read_text().strip()
        if not key:
            raise BackendError(f"missing {env_key}")
        headers = {"Content-Type": "application/json"}
        messages = [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages}
        if openai:
            headers["Authorization"] = "Bearer " + key
            messages.insert(0, {"role": "developer", "content": system})
            body["max_completion_tokens"] = MAX_TOKENS
        else:
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
            body.update(system=system, max_tokens=MAX_TOKENS)
            if self.name == "zai-anthropic":
                body["thinking"] = {"type": "disabled"}
        req = urllib.request.Request(ENDPOINTS[self.name], json.dumps(body).encode(), headers)
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            raise BackendError(f"{self.name} HTTP {exc.code}") from None
        usage = data.get("usage") or {}
        if openai:
            counts = {b: usage[a] for a, b in (("prompt_tokens", "input_tokens"),
                      ("completion_tokens", "output_tokens")) if a in usage}
            return Completion(data["choices"][0]["message"]["content"], counts)
        return Completion("".join(b["text"] for b in data["content"] if b["type"] == "text"),
                          anthropic_usage(usage))

    def _cli(self, system, prompt):
        env = os.environ.copy()
        if self.name == "claude-cli":
            env["ANTHROPIC_API_KEY"] = ""
            # Only import the subscription token when the CLI normally gets it
            # from user settings. Keep personal settings out of the experiment.
            settings = Path(env.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "settings.json"
            if not env.get("CLAUDE_CODE_OAUTH_TOKEN") and settings.is_file():
                try:
                    cfg = json.loads(settings.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    cfg = {}
                if isinstance(cfg, dict):
                    cfg_env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
                    token = (
                        cfg_env.get("CLAUDE_CODE_OAUTH_TOKEN")
                        or cfg_env.get("oauthToken")
                        or cfg.get("oauthToken")
                    )
                    if isinstance(token, str) and token.strip():
                        env["CLAUDE_CODE_OAUTH_TOKEN"] = token.strip()
            args = [self.cli or "claude", "-p", "--model", self.model, "--no-session-persistence",
                    "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--system-prompt",
                    system, "--output-format", "json"]
        else:
            args = [self.cli or "codex", "exec", "-m", self.model, "-s", "read-only", "--ephemeral",
                    "--skip-git-repo-check", "--ignore-user-config", "--disable", "shell_tool",
                    "--disable", "unified_exec", "--json", "-"]
            prompt = system + "\n\n" + prompt
        executable = shutil.which(args[0])
        if executable:
            args[0] = os.path.abspath(executable)
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(args, input=prompt, text=True, capture_output=True,
                                  cwd=cwd, env=env, timeout=self.timeout)
        if proc.returncode:
            raise BackendError(f"{self.name} exited {proc.returncode}")
        if self.name == "claude-cli":
            data = json.loads(proc.stdout)
            if isinstance(data, list):
                data = next((r for r in reversed(data) if r.get("type") == "result"), {})
            if data.get("is_error") or data.get("type") != "result":
                raise BackendError("claude-cli did not return a successful result")
            return Completion(data["result"], anthropic_usage(data.get("usage", {})))
        events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        answers = [e["item"]["text"] for e in events if e.get("type") == "item.completed"
                   and e["item"]["type"] == "agent_message"]
        if not answers or events[-1].get("type") != "turn.completed":
            raise BackendError("codex-cli did not complete the turn")
        return Completion(answers[-1], events[-1].get("usage") or {})
