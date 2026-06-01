"""Tests for benchmark.score_responses (per-response benchmark-native scoring).

Builds a tiny EES run on tmp_path with Phase 3 datapoints and Phase 4 responses,
then verifies that exact_match and bleu scoring compare the *response* against
the reference (not the reference against itself).  BERTScore is exercised only
in dry-run mode to avoid downloading a model.
"""
import json

import pytest

from benchmark.score_responses import score_run, _load_datapoint_refs, _metric_for


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _make_run(tmp_path, dp_records, resp_records, benchmark_id="squad_v2"):
    for r in dp_records:
        r.setdefault("benchmark_id", benchmark_id)
    _write_jsonl(tmp_path / "phase3_datapoints" / f"t.{benchmark_id}.datapoints.jsonl",
                 dp_records)
    _write_jsonl(tmp_path / "phase4_responses" / f"t.{benchmark_id}.s.responses.jsonl",
                 resp_records)
    return tmp_path


class TestExactMatchResponses:
    def test_correct_response_scores_one_wrong_scores_zero(self, tmp_path):
        dps = [
            {"id": "dp1", "reference_response": "Paris", "task_id": "qa"},
            {"id": "dp2", "reference_response": "London", "task_id": "qa"},
        ]
        resps = [
            {"id": "dp1__s", "datapoint_id": "dp1", "task_id": "qa",
             "student_model_id": "s", "response": "Paris"},          # correct -> 1.0
            {"id": "dp2__s", "datapoint_id": "dp2", "task_id": "qa",
             "student_model_id": "s", "response": "Berlin"},         # wrong -> 0.0
        ]
        run = _make_run(tmp_path, dps, resps)
        summary = score_run(run, metric_override="exact_match")
        assert summary["scored"] == 2
        # read back
        recs = [json.loads(l) for l in
                (run / "phase4_responses" / "t.squad_v2.s.responses.jsonl"
                 ).read_text().splitlines()]
        by_id = {r["id"]: r for r in recs}
        assert by_id["dp1__s"]["benchmark_native_score"] == 1.0
        assert by_id["dp2__s"]["benchmark_native_score"] == 0.0

    def test_not_reference_against_itself(self, tmp_path):
        # The whole point: a wrong response must NOT score 1.0
        dps = [{"id": "dp1", "reference_response": "42", "task_id": "qa"}]
        resps = [{"id": "dp1__s", "datapoint_id": "dp1", "task_id": "qa",
                  "student_model_id": "s", "response": "completely wrong"}]
        run = _make_run(tmp_path, dps, resps)
        score_run(run, metric_override="exact_match")
        rec = json.loads((run / "phase4_responses" /
                          "t.squad_v2.s.responses.jsonl").read_text().strip())
        assert rec["benchmark_native_score"] == 0.0

    def test_all_answers_accepted(self, tmp_path):
        dps = [{"id": "dp1", "reference_response": "yes",
                "_all_answers": ["yes", "affirmative"], "task_id": "qa"}]
        resps = [{"id": "dp1__s", "datapoint_id": "dp1", "task_id": "qa",
                  "student_model_id": "s", "response": "affirmative"}]
        run = _make_run(tmp_path, dps, resps)
        score_run(run, metric_override="exact_match")
        rec = json.loads((run / "phase4_responses" /
                          "t.squad_v2.s.responses.jsonl").read_text().strip())
        assert rec["benchmark_native_score"] == 1.0


class TestSidecar:
    def test_sidecar_written(self, tmp_path):
        dps = [{"id": "dp1", "reference_response": "Paris", "task_id": "qa"}]
        resps = [{"id": "dp1__s", "datapoint_id": "dp1", "task_id": "qa",
                  "student_model_id": "s", "response": "Paris"}]
        run = _make_run(tmp_path, dps, resps)
        summary = score_run(run, metric_override="exact_match")
        sidecar = run / "benchmark_response_scores.jsonl"
        assert sidecar.exists()
        rows = [json.loads(l) for l in sidecar.read_text().splitlines()]
        assert rows[0]["response_id"] == "dp1__s"
        assert rows[0]["benchmark_native_score"] == 1.0
        assert "sidecar" in summary


class TestSkipAndForce:
    def test_skips_missing_reference(self, tmp_path):
        dps = [{"id": "dp1", "reference_response": "", "task_id": "qa"}]
        resps = [{"id": "dp1__s", "datapoint_id": "dp1", "task_id": "qa",
                  "student_model_id": "s", "response": "anything"}]
        run = _make_run(tmp_path, dps, resps)
        summary = score_run(run, metric_override="exact_match")
        assert summary["skipped_no_ref"] == 1
        assert summary["scored"] == 0

    def test_idempotent_unless_force(self, tmp_path):
        dps = [{"id": "dp1", "reference_response": "Paris", "task_id": "qa"}]
        resps = [{"id": "dp1__s", "datapoint_id": "dp1", "task_id": "qa",
                  "student_model_id": "s", "response": "Paris"}]
        run = _make_run(tmp_path, dps, resps)
        score_run(run, metric_override="exact_match")
        s2 = score_run(run, metric_override="exact_match")
        assert s2["already_scored"] == 1 and s2["scored"] == 0
        s3 = score_run(run, metric_override="exact_match", force=True)
        assert s3["scored"] == 1


class TestBleuResponses:
    def test_bleu_scores_response_vs_reference(self, tmp_path):
        pytest.importorskip("nltk")
        dps = [{"id": "dp1",
                "reference_response": "the quick brown fox jumps over the lazy dog",
                "task_id": "code"}]
        resps = [{"id": "dp1__s", "datapoint_id": "dp1", "task_id": "code",
                  "student_model_id": "s",
                  "response": "the quick brown fox jumps over the lazy dog"}]
        run = _make_run(tmp_path, dps, resps, benchmark_id="codesearchnet")
        score_run(run, metric_override="bleu")
        rec = json.loads((run / "phase4_responses" /
                          "t.codesearchnet.s.responses.jsonl").read_text().strip())
        assert rec["benchmark_native_score"] > 0.9  # identical -> near 1.0


class TestHelpers:
    def test_load_refs(self, tmp_path):
        dps = [{"id": "dp1", "reference_response": "x", "task_id": "qa",
                "benchmark_id": "squad_v2"}]
        _write_jsonl(tmp_path / "phase3_datapoints" / "t.squad_v2.datapoints.jsonl", dps)
        refs = _load_datapoint_refs(tmp_path)
        assert refs["dp1"]["reference_response"] == "x"
        assert refs["dp1"]["benchmark_id"] == "squad_v2"

    def test_metric_inference(self):
        assert _metric_for("xsum", "x.jsonl", None) == "bertscore"
        assert _metric_for("squad_v2", "x.jsonl", None) == "exact_match"
        assert _metric_for(None, "foo.codesearchnet.jsonl", None) == "bleu"
        assert _metric_for("xsum", "x.jsonl", "bleu") == "bleu"  # override wins
