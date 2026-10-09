import json
import pytest
from pathlib import Path


def test_extract_accuracy_scores(frigate_module, sample_accuracy_report):
    scores, disregarded = frigate_module.extract_accuracy_scores(sample_accuracy_report, metric="f1_score")

    # Classes with tested=True and score > 0 should be extracted
    assert "dog" in scores
    assert "car" in scores
    assert "person" in scores

    # Untested classes (gt=0 or tested=False) must be disregarded
    assert "cat" not in scores
    assert "truck" not in scores
    assert "cat" in disregarded
    assert "truck" in disregarded

    assert scores["dog"] == 48.0
    assert scores["car"] == 63.2
    assert scores["person"] == 89.4


def test_compute_accuracy_balanced_quotas(frigate_module):
    scores = {"dog": 40.0, "car": 60.0, "person": 90.0}
    target_labels = ["person", "dog", "car"]
    limit = 100

    sorted_labels, quotas, deficits, health = frigate_module.compute_accuracy_balanced_quotas(
        scores=scores,
        target_labels=target_labels,
        limit=limit,
        metric="f1_score",
    )

    # Lowest score first
    assert sorted_labels[0] == "dog"
    assert sorted_labels[1] == "car"
    assert sorted_labels[2] == "person"

    # Weakest class gets highest quota
    assert quotas["dog"] > quotas["car"]
    assert quotas["car"] > quotas["person"]

    # Sum of quotas should equal limit
    assert sum(quotas.values()) == limit

    # Health status tagging
    assert health["dog"] == "WEAK"
    assert health["person"] == "GOOD"


def test_compute_cvat_balanced_quotas(frigate_module):
    # CVAT counts: person has 100, car has 10, dog has 0
    cvat_counts = {"person": 100, "car": 10, "dog": 0}
    target_labels = ["person", "car", "dog"]
    limit = 90

    sorted_labels, quotas, deficits, health = frigate_module.compute_cvat_balanced_quotas(
        cvat_counts=cvat_counts,
        target_labels=target_labels,
        limit=limit,
    )

    # Most starved class first
    assert sorted_labels[0] == "dog"
    assert sorted_labels[1] == "car"
    assert sorted_labels[2] == "person"

    assert quotas["dog"] > quotas["car"]
    assert quotas["car"] > quotas["person"]
    assert sum(quotas.values()) == limit


def test_load_and_save_processed_events(frigate_module, tmp_path):
    # Initially empty
    loaded = frigate_module.load_processed_events(base_dir=tmp_path)
    assert len(loaded) == 0

    # Save events
    events_to_save = {"event_123.456", "event_789.012"}
    frigate_module.save_processed_events(events_to_save, base_dir=tmp_path)

    # Load back
    loaded_after = frigate_module.load_processed_events(base_dir=tmp_path)
    assert loaded_after == events_to_save

    # Save overlapping batch
    frigate_module.save_processed_events({"event_789.012", "event_999.888"}, base_dir=tmp_path)
    loaded_merged = frigate_module.load_processed_events(base_dir=tmp_path)
    assert loaded_merged == {"event_123.456", "event_789.012", "event_999.888"}


def test_cli_argument_parsing(frigate_module, monkeypatch):
    test_args = [
        "frigate-to-cvat.py",
        "--limit", "150",
        "--balance-mode", "accuracy-deficit",
        "--accuracy-test-images", "50",
        "--accuracy-coco-images", "25",
        "--no-upload",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = frigate_module.parse_args()

    assert parsed.limit == 150
    assert parsed.balance_mode == "accuracy-deficit"
    assert parsed.accuracy_test_images == 50
    assert parsed.accuracy_coco_images == 25
    assert parsed.no_upload is True


@pytest.mark.parametrize("pruned_flag", [
    "--from-accuracy",
    "--accuracy-images",
    "--limit-per-label",
    "--upload",
])
def test_pruned_flags_rejected(frigate_module, monkeypatch, pruned_flag):
    monkeypatch.setattr("sys.argv", ["frigate-to-cvat.py", pruned_flag])
    with pytest.raises(SystemExit):
        frigate_module.parse_args()
