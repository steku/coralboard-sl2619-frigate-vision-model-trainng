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
