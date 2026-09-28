import pytest
from warden_laya.model import LayaModelEngine
from warden_laya.primitives import evaluate_noul, evaluate_choice, evaluate_score

@pytest.fixture
def engine():
    return LayaModelEngine(mock_mode=True)

def test_noul_calibrated_probabilities_in_unit_interval(engine):
    query = "How many weeks of parental leave do I receive?"
    candidates = [
        ("cand-001", "Employees receive 12 weeks of fully paid parental leave."),
        ("cand-002", "Public parking spaces are available on Level B2.")
    ]
    ranked = evaluate_noul(engine, query, candidates)
    assert len(ranked) == 2
    for cand_id, prob, rank in ranked:
        assert 0.0 <= prob <= 1.0
        assert isinstance(rank, int)
    # Ensure rank ordering (highest probability first)
    assert ranked[0][1] >= ranked[1][1]
    assert ranked[0][2] == 1
    assert ranked[1][2] == 2

def test_choice_confidence_distribution_sums_to_one(engine):
    query = "Can you draft a legal contract for my external freelance business?"
    choices = [
        "HR_POLICY_QUESTION",
        "COMPENSATION_INQUIRY",
        "OUT_OF_SCOPE_REQUEST",
        "DISCIPLINARY_ACTION"
    ]
    selected, conf_dist = evaluate_choice(engine, query, choices)
    assert selected in choices
    assert len(conf_dist) == 4
    prob_sum = sum(conf_dist.values())
    assert pytest.approx(prob_sum, abs=1e-5) == 1.0

def test_choice_with_fewer_than_two_choices_raises_value_error(engine):
    with pytest.raises(ValueError, match="choices must contain at least 2 distinct options"):
        evaluate_choice(engine, "query", ["ONLY_ONE"])

def test_noul_with_empty_candidates_returns_empty_list(engine):
    assert evaluate_noul(engine, "query", []) == []

def test_score_primitive_scalar_relevance(engine):
    query = "How many weeks of parental leave?"
    candidates = [
        ("cand-001", "Employees receive 12 weeks parental leave."),
        ("cand-002", "Health insurance enrollment is in November.")
    ]
    ranked = evaluate_score(engine, query, candidates)
    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]
    assert ranked[0][2] == 1
