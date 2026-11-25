"""Tests for `S3_as_of_fold` ("The Sieve") and ATProto longitudinal town replay."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bench.atproto_replay import (
    evaluate_substrates_on_town_ops,
    parse_atproto_jsonl,
    replay_external_jsonl,
    run_longitudinal_loops,
)
from substrates import AsOfAutobiographicalFoldSubstrate, AudienceTier, Event, OriginClass, TurnTrigger


class AtprotoReplayAndSieveTest(unittest.TestCase):
    def test_s3_sieve_drops_hedges_and_relations_before_entities(self) -> None:
        sub = AsOfAutobiographicalFoldSubstrate(
            head_events=2,
            tail_events=4,
            chunk_size=4,
            consolidate_factor=2,
            max_level=3,
        )
        for i in range(28):
            sub.ingest(
                Event(
                    event_id=f"ev_{i}",
                    seq=i,
                    timestamp=float(10 + i),
                    author=f"@speaker_{i % 3}.example.town",
                    origin=OriginClass.PEER_AGENT,
                    tier=AudienceTier.PUBLIC,
                    kind="post",
                    text=(
                        f"Maybe @alice.example.town -> @bob.example.town tentatively "
                        f"might dispute schema_{i} under load."
                    ),
                )
            )
        view = sub.compile_view(
            TurnTrigger(
                seq=99,
                timestamp=200.0,
                interlocutor="@alice.example.town",
                surface_tier=AudienceTier.PUBLIC,
                query_text="status of schema",
            )
        )
        labels = [b.label for b in view.prefix_blocks]
        self.assertEqual(labels[0], "HEAD")
        self.assertEqual(labels[-1], "TAIL")
        self.assertTrue(any(lbl.startswith("FOLD_L3_") for lbl in labels))

        sieve = sub.sieve_report()
        self.assertEqual(sieve["L0_tail"]["hedge_retention"], 1.0)
        self.assertEqual(sieve["L3"]["hedge_retention"], 0.0)
        self.assertGreaterEqual(sieve["L3"]["entity_retention"], 0.75)
        self.assertGreater(sieve["L3"]["entity_retention"], sieve["L3"]["relation_retention"])
        self.assertGreaterEqual(sieve["L3"]["relation_retention"], sieve["L3"]["hedge_retention"])

    def test_atproto_cid_edits_and_tombstones_separate_substrates(self) -> None:
        by_name = {r.substrate: r for r in evaluate_substrates_on_town_ops(n_threads=8, ambient_wakes=40)}
        s1 = by_name["S1_workspace_grep"]
        s2 = by_name["S2_mutable_blocks"]
        s4 = by_name["S4_dyadic_tree_zoom"]
        s6 = by_name["S6_hybrid_dyadic_cue"]

        # S1 co-retrieves stale pre-edit CIDs; S2 overwrites history destructively;
        # S4 leaks deleted leaves that were already merged into upper summary bricks;
        # S6 updates active CID recall, preserves point-in-time CID v1 audit, and retracts tombstones 100%.
        self.assertEqual(s1.cid_edit_current_acc, 0.0)
        self.assertEqual(s2.cid_edit_current_acc, 1.0)
        self.assertEqual(s2.cid_edit_audit_acc, 0.0)
        self.assertLess(s4.tombstone_retraction_acc, 1.0)

        self.assertEqual(s6.cid_edit_current_acc, 1.0)
        self.assertEqual(s6.cid_edit_audit_acc, 1.0)
        self.assertEqual(s6.tombstone_retraction_acc, 1.0)
        self.assertEqual(s6.pds_rev_staleness_acc, 1.0)

    def test_longitudinal_feedback_loops_restraint_recall_and_diary_ledger(self) -> None:
        loops = run_longitudinal_loops(wakes=60)
        # (b) Non-action parity cures compulsive action bias while preserving direct-mention replies
        self.assertLess(loops.action_only_restraint_precision, 0.35)
        self.assertEqual(loops.parity_restraint_precision, 1.0)
        self.assertEqual(loops.parity_direct_reply_recall, 1.0)

        # (c) Recall-loop tagging blocks 60-wake positive-feedback self-conditioning
        self.assertGreater(loops.untagged_final_support_count, 50)
        self.assertTrue(loops.untagged_false_promotion)
        self.assertEqual(loops.tagged_final_support_count, 1)
        self.assertFalse(loops.tagged_false_promotion)

        # (d) Progressive Sieve decay across L0 > L1 > L2 > L3 and Diary-vs-Ledger reconciliation
        self.assertGreater(loops.sieve_l0_hedge_retention, loops.sieve_l1_hedge_retention)
        self.assertGreater(loops.sieve_l1_hedge_retention, loops.sieve_l2_hedge_retention)
        self.assertGreater(loops.sieve_l2_hedge_retention, loops.sieve_l3_hedge_retention)
        self.assertEqual(loops.sieve_l3_hedge_retention, 0.0)
        self.assertGreater(loops.diary_ledger_divergence_rate, 0.40)
        self.assertEqual(loops.ledger_reconciliation_catch_rate, 1.0)

    def test_external_jetstream_jsonl_replay_cli_path(self) -> None:
        sample_records = [
            {
                "did": "did:plc:alice123",
                "time_us": 1_700_000_000_000_000,
                "kind": "commit",
                "commit": {
                    "rev": "3lbl2xa",
                    "operation": "create",
                    "collection": "town.delve.feed.post",
                    "rkey": "3k9post1",
                    "cid": "bafyreia1",
                    "record": {"$type": "town.delve.feed.post", "text": "Hello town plaza!"},
                },
            },
            {
                "did": "did:plc:alice123",
                "time_us": 1_700_000_010_000_000,
                "kind": "commit",
                "commit": {
                    "rev": "3lbl2xb",
                    "operation": "update",
                    "collection": "town.delve.feed.post",
                    "rkey": "3k9post1",
                    "cid": "bafyreia2",
                    "record": {"$type": "town.delve.feed.post", "text": "Hello town plaza (updated time)!"},
                },
            },
            {
                "did": "did:plc:bob456",
                "time_us": 1_700_000_020_000_000,
                "kind": "commit",
                "commit": {
                    "rev": "3lbl2xc",
                    "operation": "delete",
                    "collection": "town.delve.feed.post",
                    "rkey": "3k9old",
                    "cid": "",
                },
            },
        ]
        ops = parse_atproto_jsonl(json.dumps(r) for r in sample_records)
        self.assertEqual(len(ops), 3)
        self.assertEqual(ops[1].uri, "at://did:plc:alice123/town.delve.feed.post/3k9post1")
        self.assertGreater(ops[1].rev, ops[0].rev)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample_firehose.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in sample_records) + "\n", encoding="utf-8")
            summary = replay_external_jsonl(p)
            self.assertEqual(summary["input_records"], 3)
            self.assertEqual(summary["unique_dids"], 2)
            self.assertIn("S6_hybrid_dyadic_cue", summary["substrates"])


if __name__ == "__main__":
    unittest.main()
