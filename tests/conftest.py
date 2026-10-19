import importlib.util
import os
import sys
from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_module_from_path(module_name: str, file_path: Path):
    """Dynamically load python module from file path (supports hyphens in filename)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def frigate_module():
    """Import and return the frigate-to-cvat module."""
    path = REPO_ROOT / "frigate-to-cvat.py"
    return load_module_from_path("frigate_to_cvat", path)


@pytest.fixture(scope="session")
def export_cvat_module():
    """Import and return the export-cvat-to-yolo module."""
    path = REPO_ROOT / "export-cvat-to-yolo.py"
    return load_module_from_path("export_cvat_to_yolo", path)


@pytest.fixture(scope="session")
def folder_cvat_module():
    """Import and return the folder-to-cvat module."""
    path = REPO_ROOT / "folder-to-cvat.py"
    return load_module_from_path("folder_to_cvat", path)


@pytest.fixture
def sample_accuracy_report():
    """Return a mock dictionary mirroring test_server_accuracy.py benchmark output."""
    return {
        "server_url": "http://192.168.100.2:5000/v1/vision/detection",
        "split": "val",
        "images_tested": 60,
        "dataset_images_tested": 40,
        "coco_images_tested": 20,
        "overall": {
            "gt": 120,
            "pred": 115,
            "tp": 85,
            "fp": 30,
            "fn": 35,
            "precision": 73.91,
            "recall": 70.83,
            "f1_score": 72.34,
            "mean_iou": 0.742,
        },
        "per_class": {
            "dog": {
                "gt": 25,
                "pred": 20,
                "tp": 12,
                "fp": 8,
                "fn": 13,
                "precision": 60.0,
                "recall": 48.0,
                "f1_score": 48.0,
                "mean_iou": 0.65,
                "tested": True,
            },
            "car": {
                "gt": 40,
                "pred": 38,
                "tp": 24,
                "fp": 14,
                "fn": 16,
                "precision": 63.16,
                "recall": 60.0,
                "f1_score": 63.2,
                "mean_iou": 0.78,
                "tested": True,
            },
            "person": {
                "gt": 55,
                "pred": 57,
                "tp": 49,
                "fp": 8,
                "fn": 6,
                "precision": 85.96,
                "recall": 89.09,
                "f1_score": 89.4,
                "mean_iou": 0.81,
                "tested": True,
            },
            "cat": {
                "gt": 0,
                "pred": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "mean_iou": 0.0,
                "tested": False,
            },
            "truck": {
                "gt": 0,
                "pred": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "mean_iou": 0.0,
                "tested": False,
            },
        },
    }
