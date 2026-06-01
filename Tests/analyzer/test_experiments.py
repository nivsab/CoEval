"""Tests for the experiment analyzer modules (EXP-002/004/005) and stats utils.

Pure-function tests with synthetic AnalyticalUnit data and tiny fixtures.
No LLM calls, no network (rubric-overlap is exercised via its loader/cluster
math with a monkeypatched embedder to avoid downloading a model).
"""
import json
import math

import numpy as np
import pytest

from analyzer.loader import AnalyticalUnit, EESDataModel, SCORE_MAP
from analyzer import stats
from analyzer.ensemble_ablation import compute_ensemble_ablation
from analyzer.verbosity_bias import compute_verbosity_correlation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(response_id, judge, score, aspect='acc', task='task1',
          student='S1', teacher='T1', dp='dp1'):
    return AnalyticalUnit(
        response_id=response_id, datapoint_id=dp, task_id=task,
        teacher_model_id=teacher, student_model_id=student, judge_model_id=judge,
        rubric_aspect=aspect, score=score, score_norm=SCORE_MAP.get(score, float(score)
        if _is_float(score) else 0.0),
        is_self_judging=False, is_self_teaching=False,
        evaluated_at='2026-01-01T00:00:00Z',
    )


def _is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _model(units, responses=None, judges=None, tasks=None, rubrics=None, run_path='/tmp/x'):
    from pathlib import Path
    return EESDataModel(
        run_path=Path(run_path), meta={'status': 'completed'}, config={},
        rubrics=rubrics or {'task1': {'acc': 'Accuracy'}},
        datapoints={}, responses=responses or {}, eval_records=[], units=units,
        tasks=tasks or ['task1'],
        teachers=['T1'], students=sorted({u.student_model_id for u in units}),
        judges=judges or sorted({u.judge_model_id for u in units}),
        aspects_by_task={'task1': ['acc']}, target_attrs_by_task={},
        total_records=0, valid_records=0, self_judging_count=0,
        self_teaching_count=0, both_count=0, load_warnings=[], is_partial=False,
    )


# ---------------------------------------------------------------------------
# stats.py
# ---------------------------------------------------------------------------

class TestICC:
    def test_shrout_fleiss_reference(self):
        # Classic Shrout & Fleiss (1979) 6x4 matrix
        M = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
             [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
        r = stats.icc(M)
        assert r.icc_3_1 == pytest.approx(0.715, abs=0.01)
        assert r.icc_3_k == pytest.approx(0.909, abs=0.01)
        assert r.n_subjects == 6 and r.n_raters == 4

    def test_spearman_brown_identity(self):
        # ICC(3,k) must equal Spearman-Brown of ICC(3,1)
        rng = np.random.default_rng(3)
        M = rng.normal(size=(20, 3)) + rng.normal(size=(20, 1))
        r = stats.icc(M)
        k = 3
        sb = k * r.icc_3_1 / (1 + (k - 1) * r.icc_3_1)
        assert r.icc_3_k == pytest.approx(sb, abs=1e-6)

    def test_drops_nan_rows(self):
        M = [[1.0, 2.0], [3.0, 4.0], [float('nan'), 5.0]]
        r = stats.icc(M)
        assert r.n_subjects == 2

    def test_too_few(self):
        r = stats.icc([[1.0, 2.0]])
        assert math.isnan(r.icc_3_1)


class TestBootstrap:
    def test_mean_ci_brackets_point(self):
        e = stats.mean_ci(list(range(1, 101)), seed=1)
        assert e.point == pytest.approx(50.5)
        assert e.lo < e.point < e.hi
        assert e.n == 100

    def test_deterministic_with_seed(self):
        a = stats.mean_ci([1, 2, 3, 4, 5, 9, 12], seed=7)
        b = stats.mean_ci([1, 2, 3, 4, 5, 9, 12], seed=7)
        assert (a.lo, a.hi) == (b.lo, b.hi)

    def test_empty(self):
        e = stats.mean_ci([])
        assert math.isnan(e.point) and e.n == 0

    def test_single(self):
        e = stats.mean_ci([4.0])
        assert e.point == 4.0 and e.lo == 4.0 and e.hi == 4.0


class TestCorrelation:
    def test_perfect_monotone_spearman(self):
        e = stats.correlation_ci([1, 2, 3, 4, 5], [10, 20, 30, 40, 50],
                                 method='spearman', seed=1)
        assert e.point == pytest.approx(1.0)

    def test_negative_pearson(self):
        e = stats.correlation_ci([1, 2, 3, 4, 5], [5, 4, 3, 2, 1],
                                 method='pearson', seed=1)
        assert e.point == pytest.approx(-1.0)

    def test_constant_is_nan(self):
        e = stats.correlation_ci([1, 1, 1, 1], [1, 2, 3, 4], method='pearson')
        assert math.isnan(e.point)


class TestBenjaminiHochberg:
    def test_classic_rejects_prefix(self):
        rej = stats.benjamini_hochberg(
            [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205], alpha=0.05)
        assert rej[0] and rej[1]
        assert not rej[2]

    def test_nan_never_rejected(self):
        rej = stats.benjamini_hochberg([float('nan'), 0.001], alpha=0.05)
        assert rej[0] is False and rej[1] is True


# ---------------------------------------------------------------------------
# ensemble_ablation.py
# ---------------------------------------------------------------------------

class TestEnsembleAblation:
    def _two_judge_units(self):
        # 6 items, two judges agreeing perfectly -> high ICC
        units = []
        scores = ['High', 'Low', 'Medium', 'High', 'Low', 'Medium']
        for i, s in enumerate(scores):
            units.append(_unit(f'r{i}', 'J1', s))
            units.append(_unit(f'r{i}', 'J2', s))
        return units

    def test_convergence_to_full_is_one_at_k_full(self):
        m = _model(self._two_judge_units(), judges=['J1', 'J2'])
        res = compute_ensemble_ablation(m, judges=['J1', 'J2'])
        last = res['per_k'][-1]
        assert last['k'] == 2
        assert last['spearman_vs_full'] == pytest.approx(1.0)

    def test_perfect_agreement_high_icc(self):
        m = _model(self._two_judge_units(), judges=['J1', 'J2'])
        res = compute_ensemble_ablation(m, judges=['J1', 'J2'])
        k2 = res['per_k'][1]
        assert k2['icc_3_k'] > 0.95

    def test_requires_two_judges(self):
        units = [_unit('r0', 'J1', 'High')]
        m = _model(units, judges=['J1'])
        with pytest.raises(ValueError):
            compute_ensemble_ablation(m, judges=['J1'])

    def test_balanced_item_set_only(self):
        # J2 misses one item -> balanced set excludes it
        units = [_unit('r0', 'J1', 'High'), _unit('r0', 'J2', 'High'),
                 _unit('r1', 'J1', 'Low')]
        m = _model(units, judges=['J1', 'J2'])
        res = compute_ensemble_ablation(m, judges=['J1', 'J2'])
        assert res['n_items_balanced'] == 1


# ---------------------------------------------------------------------------
# verbosity_bias.py
# ---------------------------------------------------------------------------

class TestVerbosityBias:
    def test_positive_bias_detected(self):
        # longer response -> higher score for J1
        units, responses = [], {}
        for i in range(10):
            rid = f'r{i}'
            score = 'High' if i >= 5 else 'Low'
            units.append(_unit(rid, 'J1', score))
            responses[rid] = {'id': rid, 'token_count': i * 10}
        m = _model(units, responses=responses, judges=['J1'])
        res = compute_verbosity_correlation(m, judges=['J1'], method='pearson')
        assert res['judge_pooled']['J1']['point'] > 0.8

    def test_uses_stored_token_count(self):
        units = [_unit('r0', 'J1', 'High'), _unit('r1', 'J1', 'Low')]
        responses = {'r0': {'id': 'r0', 'token_count': 100},
                     'r1': {'id': 'r1', 'token_count': 5}}
        m = _model(units, responses=responses, judges=['J1'])
        res = compute_verbosity_correlation(m, judges=['J1'])
        assert res['judge_pooled']['J1']['n'] == 2

    def test_falls_back_to_response_text(self):
        # no token_count -> recompute from response text length
        units = [_unit('r0', 'J1', 'High'), _unit('r1', 'J1', 'Low'),
                 _unit('r2', 'J1', 'Medium')]
        responses = {'r0': {'id': 'r0', 'response': 'a ' * 200},
                     'r1': {'id': 'r1', 'response': 'short'},
                     'r2': {'id': 'r2', 'response': 'a ' * 50}}
        m = _model(units, responses=responses, judges=['J1'])
        res = compute_verbosity_correlation(m, judges=['J1'])
        assert res['judge_pooled']['J1']['n'] == 3

    def test_abs_reduction_key_present(self):
        units, responses = [], {}
        for i in range(8):
            rid = f'r{i}'
            units.append(_unit(rid, 'J1', 'High' if i % 2 else 'Low'))
            units.append(_unit(rid, 'J2', 'Low' if i % 2 else 'High'))
            responses[rid] = {'id': rid, 'token_count': i * 7}
        m = _model(units, responses=responses, judges=['J1', 'J2'])
        res = compute_verbosity_correlation(m, judges=['J1', 'J2'])
        assert 'abs_bias_reduction' in res
        assert 'mean_abs_individual_corr' in res


# ---------------------------------------------------------------------------
# rubric_generalization.py (loader + clustering; embedder monkeypatched)
# ---------------------------------------------------------------------------

class TestRubricOverlap:
    def test_load_and_cluster(self, tmp_path, monkeypatch):
        # Build two task rubrics with one shared criterion name
        d = tmp_path / 'phase2_rubric'
        d.mkdir()
        (d / 'taskA.rubric.json').write_text(json.dumps(
            {'accuracy': 'Is it correct', 'brevity': 'Is it short'}))
        (d / 'taskB.rubric.json').write_text(json.dumps(
            {'accuracy': 'Is it correct', 'tone': 'Is it polite'}))

        # Monkeypatch the embedder so no model download is needed.
        import analyzer.rubric_generalization as rg

        class _FakeST:
            def __init__(self, *a, **k):
                pass

            def encode(self, texts, normalize_embeddings=True):
                # accuracy texts identical -> identical vectors; others distinct
                vecs = []
                for t in texts:
                    if t.startswith('accuracy'):
                        v = np.array([1.0, 0.0, 0.0])
                    elif t.startswith('brevity'):
                        v = np.array([0.0, 1.0, 0.0])
                    else:
                        v = np.array([0.0, 0.0, 1.0])
                    vecs.append(v / np.linalg.norm(v))
                return np.array(vecs)

        monkeypatch.setattr(rg, 'SentenceTransformer', _FakeST, raising=False)
        # patch the import inside the function
        import sys
        import types
        fake_mod = types.ModuleType('sentence_transformers')
        fake_mod.SentenceTransformer = _FakeST
        monkeypatch.setitem(sys.modules, 'sentence_transformers', fake_mod)

        res = rg.compute_rubric_overlap([d], shared_min_tasks=2)
        assert res['n_criteria'] == 4
        # The two identical 'accuracy' criteria should cluster together across 2 tasks
        shared = res['shared_criteria']
        assert any(c['n_tasks'] == 2 for c in shared)

    def test_needs_two_criteria(self, tmp_path, monkeypatch):
        d = tmp_path / 'phase2_rubric'
        d.mkdir()
        (d / 'taskA.rubric.json').write_text(json.dumps({'only': 'one'}))
        import analyzer.rubric_generalization as rg
        with pytest.raises(ValueError):
            rg.compute_rubric_overlap([d])


# ---------------------------------------------------------------------------
# self_eval_control.py
# ---------------------------------------------------------------------------

class TestSelfEvalControl:
    def test_excludes_self_judging(self):
        from analyzer.self_eval_control import compute_self_eval_control, model_family
        # S1 judged by S1 (self) high, by J2 low; exclude_self_judging should drop the self unit
        units = [
            _unit('r0', 'S1', 'High', student='S1'),   # self-judge (J==S)
            _unit('r0', 'J2', 'Low', student='S1'),
            _unit('r1', 'J2', 'Low', student='S1'),
        ]
        m = _model(units, judges=['S1', 'J2'])
        m.students.append('S1')
        res = compute_self_eval_control(m)
        full = res['per_policy']['full']['S1']['mean']
        noself = res['per_policy']['exclude_self_judging']['S1']['mean']
        assert noself < full  # removing the inflated self-High lowers the mean

    def test_family_inference(self):
        from analyzer.self_eval_control import model_family
        assert model_family('gpt-4o-mini') == 'openai'
        assert model_family('claude-3.5-haiku') == 'anthropic'
        assert model_family('gemini-2.5-flash') == 'google'
        assert model_family('qwen2p5-1b5') == 'qwen'

    def test_rank_stability_keys(self):
        from analyzer.self_eval_control import compute_self_eval_control
        units = [_unit(f'r{i}', 'J1', 'High' if i % 2 else 'Low', student='S1') for i in range(4)]
        units += [_unit(f'r{i}', 'J2', 'Medium', student='S2') for i in range(4)]
        m = _model(units, judges=['J1', 'J2'])
        res = compute_self_eval_control(m)
        assert 'exclude_self_judging' in res['rank_stability']
        assert 'spearman_vs_full' in res['rank_stability']['exclude_self_judging']


class TestClusterBootstrap:
    def test_clustered_ci_wider_than_naive_with_dependence(self):
        # Build clustered data: 30 items x 3 replicates, replicates correlated within item
        import numpy as np
        from analyzer import stats
        rng = np.random.default_rng(0)
        xs, ys, cl = [], [], []
        for item in range(40):
            base = rng.normal()
            for rep in range(3):
                xs.append(base + 0.6 * rng.normal())   # moderate within-item noise
                ys.append(base + 0.6 * rng.normal())
                cl.append(item)
        clustered = stats.cluster_correlation_ci(xs, ys, cl, method='spearman', seed=1)
        assert clustered.n == 40  # n is the number of CLUSTERS, not the 120 rows
        assert clustered.lo <= clustered.hi  # valid interval ordering
        assert -1.0 <= clustered.point <= 1.0

    def test_clustered_matches_point_estimate(self):
        from analyzer import stats
        x = [1, 2, 3, 4, 5, 6]
        y = [1, 2, 3, 4, 5, 6]
        cl = [0, 0, 1, 1, 2, 2]
        e = stats.cluster_correlation_ci(x, y, cl, method='spearman', seed=1)
        assert e.point == pytest.approx(1.0)
