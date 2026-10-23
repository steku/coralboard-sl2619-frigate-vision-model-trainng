import pytest
import config_utils


def test_cvat_to_coco_id_remapping():
    # Canonical labels
    classes = ["person", "car", "dog", "custom_unknown_label"]
    task_remapping = {}
    extra_custom_classes = []

    for local_idx, cname in enumerate(classes):
        coco_id = config_utils.get_coco_id(cname)
        if coco_id is not None:
            task_remapping[local_idx] = coco_id
        else:
            if cname not in extra_custom_classes:
                extra_custom_classes.append(cname)
            task_remapping[local_idx] = 80 + extra_custom_classes.index(cname)

    assert task_remapping[0] == 0   # person
    assert task_remapping[1] == 2   # car
    assert task_remapping[2] == 16  # dog
    assert task_remapping[3] == 80  # custom_unknown_label


def test_label_line_remapping():
    # Mock label line: "1 0.5 0.5 0.2 0.3" where local id 1 -> COCO id 16 (dog)
    task_remapping = {1: 16}
    raw_line = "1 0.5 0.5 0.2 0.3"
    parts = raw_line.strip().split()
    local_id = int(parts[0])
    new_id = task_remapping.get(local_id, local_id)
    parts[0] = str(new_id)
    remapped_line = " ".join(parts)

    assert remapped_line == "16 0.5 0.5 0.2 0.3"


def test_get_latest_task_id_modes(export_cvat_module, tmp_path, monkeypatch):
    class DummyTask:
        def __init__(self, tid, name="t"):
            self.id = tid
            self.name = name

    class DummyClient:
        class tasks:
            @staticmethod
            def list():
                return [DummyTask(2), DummyTask(10), DummyTask(5)]

    monkeypatch.setattr(export_cvat_module, "STATE_FILE", tmp_path / "nonexistent_state")
    assert export_cvat_module.get_latest_task_id(DummyClient()) == 10


def test_export_and_prepare_yolo_pipeline(export_cvat_module, tmp_path, monkeypatch):
    import zipfile
    from PIL import Image

    class MockTaskObj:
        id = 7
        name = "MockExportTask"

        def export_dataset(self, format_name, filename, include_images=True):
            with zipfile.ZipFile(filename, "w") as zf:
                zf.writestr("obj.names", "person\nchicken\n")
                for i in range(4):
                    img_bytes_path = tmp_path / f"tmp_{i}.jpg"
                    Image.new("RGB", (64, 64), color="green").save(img_bytes_path)
                    zf.write(img_bytes_path, f"obj_train_data/frame_{i}.jpg")
                    cls_idx = 0 if i < 2 else 1
                    zf.writestr(f"obj_train_data/frame_{i}.txt", f"{cls_idx} 0.5 0.5 0.2 0.2\n")

    class MockClient:
        class tasks:
            @staticmethod
            def retrieve(tid):
                return MockTaskObj()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(export_cvat_module, "make_client", lambda url, credentials: MockClient())

    out_dir = tmp_path / "datasets" / "yolo"
    export_cvat_module.export_and_prepare_yolo(
        task_ids=["7"],
        output_dir=str(out_dir),
        val_split=0.25,
        seed=42,
        append=False,
        cvat_url="http://cvat.local:8080",
        cvat_user="user",
        cvat_pass="pass",
        coco_dataset="none",
        include_frigate_plus=False,
    )

    assert (out_dir / "data.yaml").is_file()
    assert (tmp_path / "datasets" / "data.yaml").is_file()

    import yaml
    with open(out_dir / "data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    assert data_cfg["names"][0] == "person"
    assert data_cfg["names"][80] == "chicken"
    train_imgs = list((out_dir / "images" / "train").glob("*.jpg"))
    val_imgs = list((out_dir / "images" / "val").glob("*.jpg"))
    assert len(train_imgs) + len(val_imgs) == 4

