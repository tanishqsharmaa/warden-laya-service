import pytest
import numpy as np
from warden_laya.model import LayaModelEngine

def test_model_engine_initialization_cpu_thread_limit():
    engine = LayaModelEngine(mock_mode=True)
    assert engine.is_loaded is True
    assert engine.num_threads == 2

def test_forward_pair_output_shape_and_range():
    engine = LayaModelEngine(mock_mode=True)
    logits = engine.forward_pair(
        query="What is the bereavement leave allowance?",
        text="Employees receive up to 5 days of paid bereavement leave."
    )
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (2,)  # Binary logits [not_relevant, relevant]

def test_forward_batch_output_shape():
    engine = LayaModelEngine(mock_mode=True)
    candidates = [
        "Chunk 1: 5 days leave.",
        "Chunk 2: Parking subsidies."
    ]
    logits = engine.forward_batch(query="leave policy", candidates=candidates)
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (2, 2)

def test_forward_empty_candidate_returns_empty():
    engine = LayaModelEngine(mock_mode=True)
    logits = engine.forward_batch(query="test query", candidates=[])
    assert logits.shape == (0, 2)

def test_forward_choice_logits_shape():
    engine = LayaModelEngine(mock_mode=True)
    choices = ["HR_POLICY", "OUT_OF_SCOPE", "COMPENSATION"]
    logits = engine.forward_choice(query="How do I get reimbursement?", choices=choices)
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (3,)
