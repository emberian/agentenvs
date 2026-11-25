"""No provider accounts or network required."""

import io
import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from bench.live_pds_sim import EPISODES, TOOLS, model_turn, parse_action, questions, simulate, summarize, thread_wake, valid_zoom
from bench.live_stream import fixture, load_stream
from bench.suite import make_substrates
from substrates import AudienceTier, SlidingWindowBareRecall, TurnTrigger
from substrates.engines import tokenize
from substrates.llm_backends import BackendError, BudgetExhausted, Completion, LLMBackend


class LiveSimulationTest(unittest.TestCase):
    def test_mirror_matches_fixture_and_withholds_removed_text(self):
        rows, deletes = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            mirror = Path(tmp)
            (mirror / "posts").mkdir()
            records = rows + [rows[0], dict(rows[0], uri="at://did:example:r0/private.dm/a")]
            (mirror / "posts/2026-10-10.jsonl").write_text("\n".join(map(json.dumps, records)))
            (mirror / "deletes.jsonl").write_text("\n".join(json.dumps(dict(uri=u, seen_at=t))
                                                          for u, t in deletes.items()))
            ops, meta = load_stream(mirror, "2026-10-10", "2026-10-20")
            offline, fixture_meta = load_stream(None, "", "")
            self.assertEqual(ops, offline)
            self.assertEqual(meta["sha256"], fixture_meta["sha256"])
            self.assertEqual(meta["operations"], {"create": 12, "update": 1, "delete": 1})
            self.assertEqual(meta["threads"], 4)
            withdrawn = next(o for o in ops if o.operation == "delete")
            predecessor = next(o for o in ops if o.uri == withdrawn.uri and o.operation == "create")
            self.assertIn("withheld", predecessor.text)
            edited = next(o for o in ops if o.operation == "update")
            self.assertIn("withheld", next(o.text for o in ops if o.uri == edited.uri))
            self.assertIn("bridge", edited.text)
            self.assertTrue(all(o.handle.startswith("@resident_") for o in ops))
            self.assertEqual(ops, sorted(ops, key=lambda o: o.timestamp))
            early, _ = load_stream(mirror, "2026-10-10", "2026-10-12")
            self.assertTrue(all(o.operation == "create" for o in early))
            self.assertIn("withheld", next(o.text for o in early if o.rkey == edited.rkey and o.did == edited.did))

    def test_cache_budget_concurrency_and_cost_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = LLMBackend("openai-http", "model-a", Path(tmp), 1)
            with patch.object(b, "_request", return_value=Completion("OK", {"output_tokens": 2})) as request:
                with ThreadPoolExecutor(3) as pool:
                    calls = list(pool.map(lambda _: b.complete("s", "p"), range(3)))
                self.assertEqual(request.call_count, 1)
                self.assertEqual(sum(not c.cached for c in calls), 1)
                with self.assertRaises(BudgetExhausted):
                    b.complete("s", "different")
            replay = LLMBackend(b.name, b.model, Path(tmp), 0)
            hit = replay.complete("s", "p")
            self.assertTrue(hit.cached)
            self.assertEqual(hit.usage["output_tokens"], 2)
            for name, model, system in ((b.name, "other", "s"), (b.name, b.model, "x"),
                                        ("codex-cli", b.model, "s")):
                with self.assertRaises(BudgetExhausted):
                    LLMBackend(name, model, Path(tmp), 0).complete(system, "p")
            trace = dict(calls=[c.__dict__ for c in calls], action="reply", tool=None, markers={})
            self.assertEqual(summarize([trace])["live_provider_tokens"], {"output_tokens": 2})

    def test_failed_attempt_counts_without_poisoning_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = LLMBackend("codex-cli", "m", Path(tmp), 1)
            with patch.object(b, "_request", side_effect=OSError("sensitive provider details")):
                with self.assertRaisesRegex(BackendError, r"^codex-cli request failed \(OSError\)$"):
                    b.complete("s", "p")
            with self.assertRaises(BudgetExhausted):
                b.complete("s", "p")
            self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_tool_result_then_a_genuine_final_turn(self):
        sub = SlidingWindowBareRecall()
        trigger = TurnTrigger(1, 1, "@neighbor", AudienceTier.PUBLIC, "Where?")
        outputs = [Completion('{"tool":"recall","arg":"dock"}', {}),
                   Completion('{"tool":"no_tool","reply":"Unknown."}', {})]
        with patch.object(LLMBackend, "complete", side_effect=outputs) as backend, patch.object(
                sub, "execute_tool", return_value="no hits") as tool:
            t = model_turn(sub, trigger, LLMBackend("codex-cli", "m"))
            tool.assert_called_once_with("recall", "dock", AudienceTier.PUBLIC)
            frame = json.loads(backend.call_args_list[1].args[1])
            self.assertEqual(frame["memory_tool_result"]["evidence"], "no hits")
            self.assertIsNone(frame["available_memory_tool"])
            self.assertEqual((t["action"], t["reply"], len(t["calls"])), ("reply", "Unknown.", 2))
        for raw in ('{"tool":"shell","arg":"read"}', 'null', '{"tool":"recall","arg":""}',
                    '{"tool":"no_tool","no_reply":true,"reply":"conflict"}'):
            with self.assertRaises(ValueError):
                parse_action(raw, "recall")
        with self.assertRaises(ValueError):
            parse_action(outputs[0].text, "recall", final=True)

    def test_all_sixteen_wakes_use_real_engines_and_record_silence(self):
        ops, _ = load_stream(None, "", "")
        for sub in make_substrates():
            if sub.name.split("_")[0] not in TOOLS:
                continue
            with self.subTest(sub=sub.name), patch.object(
                    LLMBackend, "complete", return_value=Completion('{"tool":"no_tool","no_reply":true}', {})):
                traces = simulate(type(sub), LLMBackend("claude-cli", "m"), ops, 16)
                self.assertEqual([t["episode"] for t in traces], [n for n in range(1, 5) for _ in range(4)])
                self.assertTrue(all(t["action"] == "no_reply" and len(t["calls"]) == 1 for t in traces))
        for e in EPISODES:
            text = f"Lost badges at dusk. Latecomers followed a {e.beacon} suspended above our stall."
            self.assertFalse(set(tokenize(text)) & set(tokenize(questions(e)[1])))

    def test_dyadic_references_and_cross_day_holdout(self):
        self.assertTrue(valid_zoom("8+4", 12))
        self.assertTrue(valid_zoom("1+1", 2))
        for ref in ("8+18", "1+4", "0+0", "0+32", "leaf"):
            self.assertFalse(valid_zoom(ref, 16))
        ops, _ = load_stream(None, "", "")
        with patch.object(LLMBackend, "complete", return_value=Completion('{"tool":"no_tool","no_reply":true}', {})) as provider:
            t = thread_wake(SlidingWindowBareRecall(), LLMBackend("claude-cli", "m"), ops)
            frame = json.loads(provider.call_args.args[1])
            self.assertLess(t["parent_day"], t["reply_day"])
            self.assertEqual(t["prior_operations"], 1)
            self.assertNotIn(t["query"], json.dumps(frame["memory"]))
            self.assertEqual(t["query"], frame["incoming_public_post"])

    def test_http_transports_and_codex_event_stream(self):
        shapes = {
            "anthropic-http": {"content": [{"type": "thinking"}, {"type": "text", "text": "ok"}],
                               "usage": {"output_tokens": 2}},
            "openai-http": {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 2}},
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "OPENAI_API_KEY": "unused", "ANTHROPIC_API_KEY": "unused"}):
            for name, response in shapes.items():
                with self.subTest(name=name), patch("urllib.request.build_opener") as opener:
                    opener.return_value.open.return_value = io.BytesIO(json.dumps(response).encode())
                    result = LLMBackend(name, "m", Path(tmp)).complete("system", "prompt")
                    self.assertEqual((result.text, result.usage), ("ok", {"output_tokens": 2}))
                    req = opener.return_value.open.call_args.args[0]
                    self.assertEqual(json.loads(req.data)["messages"][-1]["content"], "prompt")
                    self.assertNotIn("unused", req.full_url)
            rows = [{"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
                    {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 1}}]
            proc = subprocess.CompletedProcess([], 0, "\n".join(map(json.dumps, rows)), "")
            with patch("subprocess.run", return_value=proc) as cli:
                result = LLMBackend("codex-cli", "m", Path(tmp)).complete("system", "prompt")
                self.assertEqual((result.text, result.usage["output_tokens"]), ("ok", 1))
                self.assertIn("read-only", cli.call_args.args[0])
                self.assertIn("--ephemeral", cli.call_args.args[0])

    def test_subscription_cli_usage_and_error_handling(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "ANTHROPIC_API_KEY": "unused", "CLAUDE_CONFIG_DIR": tmp}):
            b = LLMBackend("claude-cli", "m", Path(tmp))
            done = {"type": "result", "result": "yes", "usage": {"input_tokens": 7}}
            proc = subprocess.CompletedProcess([], 0, json.dumps([{"type": "system"}, done]), "")
            with patch("subprocess.run", return_value=proc) as call:
                self.assertEqual(b.complete("s", "p").usage, {"input_tokens": 7})
                self.assertEqual(call.call_args.kwargs["env"]["ANTHROPIC_API_KEY"], "")
                self.assertIn("--no-session-persistence", call.call_args.args[0])
                proc.stdout = json.dumps(dict(done, is_error=True))
                with self.assertRaises(BackendError):
                    b.complete("s", "p2")


if __name__ == "__main__":
    unittest.main()
