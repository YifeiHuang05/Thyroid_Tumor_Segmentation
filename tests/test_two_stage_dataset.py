from two_stage_dataset import infer_patient_label, make_stratified_split


def test_infer_patient_label_uses_required_stage_mapping():
    assert infer_patient_label({"nodule_1": {"Histopathology": "Papillary thyroid carcinoma"}}) == 0
    assert infer_patient_label({"nodule_1": {"Histopathology": "Follicular thyroid carcinoma"}}) == 1
    assert infer_patient_label({"nodule_1": {"Histopathology": "Thyroid cyst disease"}}) == 2


def test_make_stratified_split_keeps_patient_classes_disjoint_and_balanced():
    patient_records = [
        {"patient_id": "p1", "label": 0},
        {"patient_id": "p2", "label": 0},
        {"patient_id": "p3", "label": 1},
        {"patient_id": "p4", "label": 1},
        {"patient_id": "p5", "label": 2},
        {"patient_id": "p6", "label": 2},
    ]

    train, val, test = make_stratified_split(patient_records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert set(train) | set(val) | set(test) == {"p1", "p2", "p3", "p4", "p5", "p6"}
    assert set(train).isdisjoint(set(val))
    assert set(val).isdisjoint(set(test))
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(val) + len(test) == 6
