"""Unit tests for closed-loop longitudinal drift, succession replay, XRPC loader, and repo hygiene."""

from __future__ import annotations

import json
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from bench.longitudinal_drift import run_experiment
from bench.wesvet_replay import evaluate_deterministic_succession, evaluate_live_succession
from substrates.xrpc_loader import (
    DELVE_POST_COLLECTION,
    XrpcError,
    list_pds_repos,
    load_actor_events,
    load_pds_town_threads,
    resolve_actor,
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
