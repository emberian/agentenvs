"""Unit tests for closed-loop longitudinal drift, succession replay, XRPC loader, and repo hygiene."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from bench.longitudinal_drift import LongitudinalLLM, run_experiment
from bench.wesvet_replay import (
    ExtractiveDyadicBrickSubstrate,
    evaluate_deterministic_succession,
    evaluate_live_succession,
    main as wesvet_main,
    scores_header_decoy_chase,
)
from substrates import AudienceTier, Event, OriginClass, TurnTrigger
from substrates.llm_backends import Completion
from substrates.xrpc_loader import (
    DELVE_POST_COLLECTION,
    ResolvedIdentity,
    XrpcError,
    _validate_endpoint_url,
    list_pds_repos,
    list_repo_records,
    load_actor_events,
    load_pds_town_threads,
    records_to_events,
    resolve_actor,
    resolve_did_document,
)


class _MockAtprotoHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        host = f"http://127.0.0.1:{self.server.server_port}"
        if self.path.startswith("/xrpc/com.atproto.identity.resolveHandle"):
            self._send_json(200, {"did": "did:plc:mockresident01"})
            return
        if self.path == "/did:plc:mockresident01":
            self._send_json(
                200,
                {
                    "id": "did:plc:mockresident01",
                    "alsoKnownAs": ["at://archivist.example.town"],
                    "service": [
                        {
                            "id": "#atproto_pds",
                            "type": "AtprotoPersonalDataServer",
                            "serviceEndpoint": host,
                        }
                    ],
                },
            )
            return
        if self.path.startswith("/xrpc/com.atproto.sync.listRepos"):
            self._send_json(
                200,
                {
                    "repos": [
                        {"did": "did:plc:mockresident01", "head": "bafy_h1", "rev": "3lbl2xa", "active": True},
                        {"did": "did:plc:mockresident02", "head": "bafy_h2", "rev": "3lbl2zz", "active": True},
                    ]
                },
            )
            return
        if self.path.startswith("/xrpc/com.atproto.repo.listRecords") and "collection=town.delve.feed.post" in self.path:
            if "repo=did%3Aplc%3Amockresident01" in self.path:
                self._send_json(
                    200,
                    {
                        "records": [
                            {
                                "uri": "at://did:plc:mockresident01/town.delve.feed.post/3k9root",
                                "cid": "bafy_delve_root",
                                "value": {
                                    "$type": "town.delve.feed.post",
                                    "text": "Opening town deliberation on lexicon schemas.",
                                    "createdAt": "2026-10-15T09:00:00Z",
                                },
                            }
                        ]
                    },
                )
            else:
                self._send_json(
                    200,
                    {
                        "records": [
                            {
                                "uri": "at://did:plc:mockresident02/town.delve.feed.post/3k9reply",
                                "cid": "bafy_delve_reply",
                                "value": {
                                    "$type": "town.delve.feed.post",
                                    "text": "Replying with rev-stamped counter-proposal.",
                                    "createdAt": "2026-10-15T09:15:00Z",
                                    "reply": {
                                        "root": {"uri": "at://did:plc:mockresident01/town.delve.feed.post/3k9root"},
                                        "parent": {"uri": "at://did:plc:mockresident01/town.delve.feed.post/3k9root"},
                                    },
                                },
                            }
                        ]
                    },
                )
            return
        if self.path.startswith("/xrpc/com.atproto.repo.listRecords"):
            if "cursor=page2" in self.path:
                self._send_json(
                    200,
                    {
                        "records": [
                            {
                                "uri": "at://did:plc:mockresident01/app.bsky.feed.post/3k9001",
                                "cid": "bafy_p1",
                                "value": {
                                    "$type": "app.bsky.feed.post",
                                    "text": "Earlier post with @alice.example mention.",
                                    "createdAt": "2026-10-10T12:00:00Z",
                                    "facets": [{"features": [{"did": "did:plc:alice99"}]}],
                                },
                            }
                        ]
                    },
                )
            else:
                self._send_json(
                    200,
                    {
                        "cursor": "page2",
                        "records": [
                            {
                                "uri": "at://did:plc:mockresident01/app.bsky.feed.post/3k9002",
                                "cid": "bafy_p2",
                                "value": {
                                    "$type": "app.bsky.feed.post",
                                    "text": "Newer post on the public PDS.",
                                    "createdAt": "2026-10-11T12:00:00Z",
                                },
                            }
                        ],
                    },
                )
            return
        self._send_json(404, {"error": "not_found"})

    def _send_json(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class LongitudinalWesvetXrpcTest(unittest.TestCase):
    def test_closed_loop_40_wake_feedback_separates_substrates(self) -> None:
        report = run_experiment(
            backend="mock",
            model="deterministic-mock",
            wakes=40,
            mirror_dir=None,
            cache_dir=Path("/tmp/agentenvs_mock_unused"),
            jobs=2,
        )
        arms = report["arms"]
        l1 = arms["L1_SelfEditedBlocks"]
        l2 = arms["L2_RecursiveAsOfFold"]
        l4 = arms["L4_ImmutableLedgerHybrid"]

        # Closed-loop state feedback: L1 rewrites <persona> after soft pushback and loses stance fidelity;
        # L2 drops hedges across L2 consolidation; L4 preserves persona, caveats, and withdrawn tombstone!
        self.assertGreater(l1["persona_block_drift"], 0.25)
        self.assertGreater(l1["soft_pushback_yield_rate"], l1["hard_pushback_yield_rate"])
        self.assertLess(l2["epistemic_humility_rate"], l4["epistemic_humility_rate"])

        self.assertEqual(l4["soft_pushback_yield_rate"], 0.0)
        self.assertEqual(l4["hard_pushback_yield_rate"], 0.0)
        self.assertEqual(l4["epistemic_humility_rate"], 1.0)
        self.assertEqual(l4["withdrawn_commitment_respect_rate"], 1.0)
        self.assertEqual(l4["private_leak_count"], 0)
        self.assertGreater(l4["final_pr"], l1["final_pr"])

        # Per-arm cost attribution and actor-prompt prefix reuse are reported for every arm.
        for arm in arms.values():
            cost = arm["cost"]
            self.assertEqual(cost["actor_calls"], 40)
            self.assertGreaterEqual(cost["total_input_tokens"], cost["actor_input_tokens"])
            self.assertGreaterEqual(arm["kv_prefix_reuse_rate"], 0.0)
            self.assertLessEqual(arm["kv_prefix_reuse_rate"], 1.0)
        # Pull-only and immutable-ledger arms make no background maintenance calls;
        # self-edited blocks pay one curator call per wake.
        self.assertEqual(arms["L0_PullOnly"]["cost"]["maintenance_calls"], 0)
        self.assertEqual(l4["cost"]["maintenance_calls"], 0)
        self.assertEqual(l1["cost"]["maintenance_calls"], 40)

    def test_wesvet_succession_and_extractive_dyadic_bricks(self) -> None:
        by_name = {r.substrate: r for r in evaluate_deterministic_succession()}
        essay = by_name["E_ContinuationEssay"]
        bricks = by_name["E_ExtractiveDyadicBricks"]

        # Hand-written continuation loses granular paths/nogoods and carries poisoned prose hypothesis;
        # ExtractiveDyadicBricks preserves 100% of specs/nogoods/reversals + high KV prefix stability.
        self.assertLess(essay.fact_and_path_recall, 0.50)
        self.assertEqual(essay.escalation_localization_clean, 0.0)

        self.assertEqual(bricks.fact_and_path_recall, 1.0)
        self.assertEqual(bricks.arch_reversal_fidelity, 1.0)
        self.assertEqual(bricks.nogood_retention_rate, 1.0)
        self.assertEqual(bricks.dirty_and_daemon_accuracy, 1.0)
        self.assertEqual(bricks.escalation_localization_clean, 1.0)
        self.assertGreater(bricks.kv_prefix_block_hit_rate, 0.65)

        mock_llm_res = evaluate_live_succession(mock_llm=True)
        self.assertEqual(
            mock_llm_res["E_ContinuationEssay_ProseHyp"]["stage10_escalation_localization"],
            "header.rs (poisoned decoy)",
        )
        self.assertEqual(
            mock_llm_res["E_ExtractiveBricks_FactsOnly"]["stage10_escalation_localization"],
            "varint.rs (correct)",
        )

    def test_xrpc_loader_resolves_paginates_and_anonymizes_over_loopback_pds(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _MockAtprotoHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            identity = resolve_actor(
                "archivist.example.town",
                service_url=base,
                plc_directory_url=base,
                allow_loopback_http=True,
            )
            self.assertEqual(identity.did, "did:plc:mockresident01")
            self.assertEqual(identity.handle, "@archivist.example.town")

            _, events = load_actor_events(
                "archivist.example.town",
                limit=10,
                max_pages=3,
                service_url=base,
                plc_directory_url=base,
                anonymize=True,
                allow_loopback_http=True,
            )
            self.assertEqual(len(events), 2)
            # Chronologically ordered (2026-10-10 before 2026-10-11) and anonymized in memory
            self.assertLess(events[0].timestamp, events[1].timestamp)
            self.assertEqual(events[0].author, "@res_001")
            self.assertNotIn("alice.example", events[0].text)
            self.assertIn("@peer_ref", events[0].text)

            # Multi-repo PDS discovery (`listRepos`) + `town.delve.feed.post` thread assembly
            repos = list_pds_repos(base, limit=10, allow_loopback_http=True)
            self.assertEqual(len(repos), 2)
            self.assertEqual(repos[0].rev, "3lbl2xa")

            town_events, threads = load_pds_town_threads(
                base,
                collection=DELVE_POST_COLLECTION,
                max_repos=10,
                per_repo_limit=10,
                anonymize=True,
                allow_loopback_http=True,
            )
            self.assertEqual(len(town_events), 2)
            self.assertEqual(len(threads), 1)
            only_thread = next(iter(threads.values()))
            self.assertEqual(len(only_thread), 2)
            self.assertIn("reply_to=", town_events[1].text)

            # Non-HTTPS rejected when loopback flag is disabled
            with self.assertRaises(XrpcError):
                resolve_actor("archivist.example.town", service_url=base, plc_directory_url=base)
        finally:
            server.shutdown()
            server.server_close()

    def test_wesvet_cli_flags_budget_and_negation_hygiene(self) -> None:
        # #tooth A1: --max-calls 0 raises RuntimeError("Live call budget exhausted") on cache miss,
        # and succeeds with 0 network calls when all prompts are already cached.
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "wesvet_cache"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Live call budget exhausted"):
                wesvet_main(
                    [
                        "--live",
                        "--backend",
                        "zai-anthropic",
                        "--model",
                        "glm-5.3",
                        "--cache-dir",
                        str(cache_dir),
                        "--max-calls",
                        "0",
                    ]
                )

            def _fake_open(req, timeout=90):
                body = {"content": [{"type": "text", "text": "crates/index/src/varint.rs APPEND_SEGMENT_INDEX_V2"}], "usage": {"input_tokens": 11, "output_tokens": 5}}
                return io.BytesIO(json.dumps(body).encode("utf-8"))

            with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}), patch(
                "urllib.request.OpenerDirector.open", side_effect=_fake_open
            ) as net_open:
                rc_warm = wesvet_main(
                    [
                        "--live",
                        "--backend",
                        "zai-anthropic",
                        "--model",
                        "glm-5.3",
                        "--cache-dir",
                        str(cache_dir),
                        "--max-calls",
                        "60",
                    ]
                )
                self.assertEqual(rc_warm, 0)
                self.assertGreater(net_open.call_count, 0)

            with contextlib.redirect_stdout(io.StringIO()), patch(
                "urllib.request.OpenerDirector.open", side_effect=AssertionError("unexpected network call")
            ):
                rc_replay = wesvet_main(
                    [
                        "--live",
                        "--backend",
                        "zai-anthropic",
                        "--model",
                        "glm-5.3",
                        "--cache-dir",
                        str(cache_dir),
                        "--max-calls",
                        "0",
                    ]
                )
                self.assertEqual(rc_replay, 0)

        # #tooth A4: negated warnings about header.rs are not misclassified as decoy chases
        self.assertFalse(scores_header_decoy_chase("The endianness theory in crates/store/src/header.rs is unverified — don't start by rewriting header.rs."))
        self.assertFalse(scores_header_decoy_chase("Do not rewrite crates/store/src/header.rs without evidence."))
        self.assertFalse(scores_header_decoy_chase("We are not rewriting header.rs and should avoid rewriting header.rs."))
        self.assertTrue(scores_header_decoy_chase("Root cause is endianness inside crates/store/src/header.rs; rewrite header.rs first."))

        # #tooth A4: facts_only_escalation=False preserves extract_technical_anchors invariants while retaining PROSE_HYPOTHESIS:
        with_hyp = ExtractiveDyadicBrickSubstrate(high_bytes=4096, facts_only_escalation=False)
        facts_only = ExtractiveDyadicBrickSubstrate(high_bytes=4096, facts_only_escalation=True)
        owner_ev = Event("e_own", 1, 1.0, "operator", OriginClass.OWNER, AudienceTier.PUBLIC, "operator", "SPEC:WAL_CORE in crates/store/src/wal.rs")
        brief_ev = Event("e_brf", 2, 2.0, "@coord", OriginClass.PEER_AGENT, AudienceTier.PUBLIC, "post", "COORDINATOR_BRIEF: daemon still running live")
        esc_ev = Event(
            "e_esc",
            3,
            3.0,
            "seat_3",
            OriginClass.SELF,
            AudienceTier.PUBLIC,
            "post",
            "FACT:panics_at_shift_35 PROSE_HYPOTHESIS: rewrite crates/store/src/header.rs!",
        )
        for ev in (owner_ev, brief_ev, esc_ev):
            with_hyp.ingest(ev)
            facts_only.ingest(ev)
        trig = TurnTrigger(4, 4.0, "operator", AudienceTier.PUBLIC, "status of shift_35 escalation")
        v_hyp = "\n".join([b.text for b in with_hyp.compile_view(trig).prefix_blocks] + [c.summary_line for c in with_hyp.compile_view(trig).tail_cues])
        v_facts = "\n".join([b.text for b in facts_only.compile_view(trig).prefix_blocks] + [c.summary_line for c in facts_only.compile_view(trig).tail_cues])
        self.assertIn("OWNER_SPEC", v_hyp)
        self.assertIn("UNVERIFIED_PARAPHRASE", v_hyp)
        self.assertNotIn("still running live", v_hyp)
        self.assertIn("PROSE_HYPOTHESIS:", v_hyp)
        self.assertNotIn("PROSE_HYPOTHESIS:", v_facts)

    def test_xrpc_ssrf_did_web_pagination_rev_and_anonymize_hardening(self) -> None:
        # #tooth A2: reject encoded path/query injection in did:web and non-public/malformed base endpoints
        with self.assertRaises(XrpcError):
            resolve_did_document("did:web:evil.example%2Fadmin%3Fx%3D1")
        with self.assertRaises(XrpcError):
            resolve_did_document("did:web:example.com:..:secret")
        for bad_url in (
            "https://127.0.0.1:8443",
            "https://10.0.0.5",
            "https://169.254.169.254",
            "https://localhost:8443",
            "https://svc.internal",
            "https://user:pass@example.com",
            "https://example.com/unexpected/base",
            "https://example.com?leak=1",
            "https://example.com#frag",
        ):
            with self.assertRaises(XrpcError, msg=bad_url):
                _validate_endpoint_url(bad_url, allow_loopback_http=False)

        # #tooth A2: W3C did:web subpaths resolve to /<seg1>/<seg2>/did.json
        with patch("substrates.xrpc_loader._get_json", return_value={"id": "did:web:example.com:user:alice"}) as get_json:
            doc = resolve_did_document("did:web:example.com:user:alice")
            self.assertEqual(doc["id"], "did:web:example.com:user:alice")
            get_json.assert_called_once_with(
                "https://example.com/user/alice/did.json",
                timeout=15.0,
                allow_loopback_http=False,
            )

        # #tooth A6: active pre-filter before limit, dynamic max_pages, monotonic rkey rev, and unified scrubbing
        repo_pages = [
            {
                "cursor": "c2",
                "repos": [
                    {"did": "did:plc:inactive001", "head": "h0", "rev": "3zzzzzz", "active": False},
                    {"did": "did:plc:active001", "head": "h1", "rev": "3zzzzzz", "active": True},
                ],
            },
            {
                "repos": [
                    {"did": "did:plc:active002", "head": "h2", "rev": "3zzzzzz", "active": True},
                ],
            },
        ]
        with patch("substrates.xrpc_loader._get_json", side_effect=repo_pages):
            active_repos = list_pds_repos("https://pds.delve.town", limit=2, max_pages=2)
            self.assertEqual([r.did for r in active_repos], ["did:plc:active001", "did:plc:active002"])

        # Dynamic max_pages = max(1, (250 + 99) // 100) == 3 pages when max_pages is omitted
        rec_pages = [
            {
                "cursor": f"p{page_idx + 1}",
                "records": [
                    {
                        "uri": f"at://did:plc:active001/town.delve.feed.post/3k9{page_idx:02d}{i:02d}",
                        "cid": f"cid_{page_idx}_{i}",
                        "value": {
                            "text": f"See https://secret.example/p/{i} at://alice.bsky.social/app.bsky.feed.post/3k1 contact_me@domain.org @alice.example",
                            "createdAt": f"2026-10-15T10:{page_idx:02d}:{i:02d}Z",
                        },
                    }
                    for i in range(2)
                ],
            }
            for page_idx in range(3)
        ]
        with patch("substrates.xrpc_loader._get_json", side_effect=rec_pages) as get_recs:
            fetched = list_repo_records("https://pds.delve.town", "did:plc:active001", limit=250)
            self.assertEqual(get_recs.call_count, 3)
            self.assertEqual(len(fetched), 6)

        ident = ResolvedIdentity("did:plc:active001", "@active001.example", "https://pds.delve.town")
        single_evs = records_to_events(rec_pages[0]["records"], ident, anonymize=True)
        for txt in (single_evs[0].text,):
            self.assertNotIn("secret.example", txt)
            self.assertNotIn("alice.bsky.social", txt)
            self.assertNotIn("contact_me", txt)
            self.assertNotIn("domain.org", txt)
        self.assertLess(
            single_evs[0].subject_revs["did:plc:res_001"],
            single_evs[1].subject_revs["did:plc:res_001"],
        )

        with patch(
            "substrates.xrpc_loader.list_pds_repos",
            return_value=[active_repos[0]],
        ), patch("substrates.xrpc_loader.list_repo_records", return_value=rec_pages[0]["records"]):
            town_evs, _ = load_pds_town_threads("https://pds.delve.town", anonymize=True)
            self.assertEqual(len(town_evs), 2)
            self.assertLess(
                town_evs[0].subject_revs["did:plc:res_001"],
                town_evs[1].subject_revs["did:plc:res_001"],
            )
            self.assertNotIn("secret.example", town_evs[0].text)
            self.assertNotIn("contact_me", town_evs[0].text)
            self.assertNotIn("alice.bsky.social", town_evs[0].text)

    def test_longitudinal_oauth_settings_fallback_and_cached_token_totals(self) -> None:
        # #tooth A5: claude-cli reads oauthToken from CLAUDE_CONFIG_DIR/settings.json while blanking ANTHROPIC_API_KEY,
        # and --max-calls 0 cache replays populate top-level cached_input_tokens / cached_output_tokens.
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp) / "claude_cfg"
            cfg_dir.mkdir()
            (cfg_dir / "settings.json").write_text(json.dumps({"env": {"oauthToken": "oauth-test-tok"}}), encoding="utf-8")
            cache_dir = Path(tmp) / "long_cache"

            llm = LongitudinalLLM(backend="claude-cli", model="claude-haiku-4-5", cache_dir=cache_dir, max_calls=2)
            fake_out = json.dumps({"type": "result", "is_error": False, "result": "unverified append-only reply", "usage": {"input_tokens": 17, "output_tokens": 9}})
            fake_proc = subprocess.CompletedProcess(["claude"], 0, fake_out, "")
            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(cfg_dir), "ANTHROPIC_API_KEY": "should-blank"}, clear=False):
                os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
                with patch("subprocess.run", return_value=fake_proc) as run_mock:
                    comp = llm.complete("actor", "sys", "prompt-1")
                    self.assertFalse(comp.cached)
                    passed_env = run_mock.call_args.kwargs["env"]
                    self.assertEqual(passed_env["ANTHROPIC_API_KEY"], "")
                    self.assertEqual(passed_env["CLAUDE_CODE_OAUTH_TOKEN"], "oauth-test-tok")

            # Warm 1-wake run then replay with max_calls=0 and verify cached token totals > 0
            def _stub_live(self_llm, system: str, prompt: str) -> Completion:
                return Completion(text="Preliminary unverified append-only check.", usage={"input_tokens": 25, "output_tokens": 10})

            with patch.object(LongitudinalLLM, "_live_request", _stub_live):
                warm = run_experiment(
                    backend="zai-anthropic",
                    model="glm-5.3",
                    wakes=1,
                    mirror_dir=None,
                    cache_dir=cache_dir,
                    max_calls=20,
                    jobs=1,
                )
                self.assertGreater(warm["live_calls"], 0)
                self.assertGreater(warm["input_tokens"], 0)

            replay = run_experiment(
                backend="zai-anthropic",
                model="glm-5.3",
                wakes=1,
                mirror_dir=None,
                cache_dir=cache_dir,
                max_calls=0,
                jobs=1,
            )
            self.assertEqual(replay["live_calls"], 0)
            self.assertGreater(replay["cached_hits"], 0)
            self.assertGreater(replay["cached_input_tokens"], 0)
            self.assertGreater(replay["cached_output_tokens"], 0)

    def test_repository_tracked_files_contain_no_local_home_paths_or_trailers(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked = [repo_root / rel for rel in proc.stdout.splitlines() if rel.strip()]
        # Include newly created untracked deliverables in the workspace check too
        candidates = set(tracked) | set(repo_root.glob("**/*.py")) | set(repo_root.glob("**/*.md"))
        home_prefix = "/" + "Users/"
        trailer_prefix = "Co-" + "Authored-By:"
        for path in sorted(candidates):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(repo_root).parts):
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(home_prefix, text, f"local home path leaked in {path}")
            self.assertNotIn(trailer_prefix, text, f"model trailer leaked in {path}")


if __name__ == "__main__":
    unittest.main()
