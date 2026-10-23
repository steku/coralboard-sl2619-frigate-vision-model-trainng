#!/usr/bin/env python3
"""
Unit tests for report-cvat-balance.py
"""

import pytest


def test_create_progress_bar(report_balance_module):
    bar_half = report_balance_module.create_progress_bar(50.0, width=20)
    assert len(bar_half) == 20
    assert bar_half.count("█") == 10
    assert bar_half.count("░") == 10

    bar_empty = report_balance_module.create_progress_bar(0.0, width=10)
    assert bar_empty == "░" * 10


def test_fetch_cvat_data_and_print_report(report_balance_module, monkeypatch, capsys):
    class MockResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class MockSession:
        def __init__(self):
            self.auth = None

        def get(self, url, timeout=10):
            if "/api/tasks?" in url:
                return MockResponse({
                    "results": [
                        {"id": 1, "name": "Task 1", "mode": "annotation", "status": "completed", "size": 25},
                        {"id": 2, "name": "Task 2", "mode": "annotation", "status": "annotation", "size": 15},
                    ]
                })
            elif "/api/labels?" in url:
                return MockResponse({
                    "results": [
                        {"id": 10, "name": "person"},
                        {"id": 11, "name": "dog"},
                    ]
                })
            elif "/api/tasks/1/annotations" in url:
                return MockResponse({
                    "shapes": [{"label_id": 10}] * 120 + [{"label_id": 11}] * 5,
                    "tracks": [],
                    "tags": [],
                })
            elif "/api/tasks/2/annotations" in url:
                return MockResponse({
                    "shapes": [{"label_id": 10}] * 10,
                    "tracks": [{"label_id": 11}] * 2,
                    "tags": [],
                })
            elif "/api/jobs?" in url:
                return MockResponse({
                    "results": [
                        {"id": 101, "state": "new", "stage": "annotation", "status": "annotation"}
                    ]
                })
            elif "/api/jobs/101/annotations" in url:
                return MockResponse({
                    "shapes": [{"label_id": 10}] * 5,
                    "tracks": [],
                    "tags": [],
                })
            return MockResponse({})

    import requests
    monkeypatch.setattr(requests, "Session", MockSession)

    data = report_balance_module.fetch_cvat_data(
        base_url="http://cvat.local:8080",
        auth=("user", "pass"),
        target_task_ids=["1"],
        include_jobs=True,
    )

    assert data["total_tasks"] == 1
    assert data["total_frames"] == 25
    assert data["total_annotations"] == 125
    assert data["overall_class_counts"]["person"] == 120
    assert data["overall_class_counts"]["dog"] == 5
    assert len(data["tasks"][0]["jobs"]) == 1

    report_balance_module.print_text_report(data, include_jobs=True)
    captured = capsys.readouterr()
    assert "CVAT DATASET OBJECT BALANCE AUDIT REPORT" in captured.out
    assert "Task #1:" in captured.out
    assert "Task 1" in captured.out
    assert "dog" in captured.out
    assert "CRITICAL" in captured.out
