from __future__ import annotations

import torch

from train_v4_reverse_state_curriculum import (
    sample_curriculum_severity,
    validate_curriculum,
)


def test_curriculum_probabilities_are_normalized_without_changing_severities():
    severities, probabilities = validate_curriculum(
        [0.5, 1.0, 2.0],
        [2.0, 4.0, 4.0],
    )
    assert severities == (0.5, 1.0, 2.0)
    assert probabilities == (0.2, 0.4, 0.4)
    assert abs(sum(probabilities) - 1.0) < 1e-8


def test_curriculum_sampling_is_reproducible_and_uses_only_declared_values():
    severities = [0.5, 1.0, 2.0]
    probabilities = [0.2, 0.4, 0.4]

    g1 = torch.Generator(device="cpu")
    g2 = torch.Generator(device="cpu")
    g1.manual_seed(1234)
    g2.manual_seed(1234)

    seq1 = [
        sample_curriculum_severity(
            severities,
            probabilities,
            generator=g1,
        )
        for _ in range(128)
    ]
    seq2 = [
        sample_curriculum_severity(
            severities,
            probabilities,
            generator=g2,
        )
        for _ in range(128)
    ]

    assert seq1 == seq2
    assert set(seq1).issubset(set(severities))
    # All three severities should appear for a representative deterministic draw.
    assert set(seq1) == set(severities)
