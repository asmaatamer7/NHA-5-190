"""Run: python -m pytest tests/test_m2_resnet.py -q
Builder tests need no torch. Model tests skip automatically if torch is not installed."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TAX = ROOT / "configs" / "m1" / "taxonomy.json"


def _make(tmp_path):
    tax = json.loads(TAX.read_text(encoding="utf-8"))
    pv = list(tax["sources"]["polyvore_outfits"]["source_id_to_semantic"])[:5]
    man, spl = [], []
    for o in range(30):
        for k in range(3):
            i = f"pv_{o}_{k}"
            man.append(
                {
                    "item_id": i,
                    "source_dataset": "Polyvore Outfits",
                    "category_ids": pv[(o + k) % 5],
                    "original_image_reference": f"p/{i}.jpg",
                }
            )
            spl.append({"item_id": i, "target_split": "validation", "group_id": f"{o}:{k}"})
    pd.DataFrame(man).to_csv(tmp_path / "m.csv", index=False)
    pd.DataFrame(spl).to_csv(tmp_path / "s.csv", index=False)


def _run(tmp_path, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_classification_manifest.py"),
            "--manifest",
            str(tmp_path / "m.csv"),
            "--splits",
            str(tmp_path / "s.csv"),
            "--taxonomy",
            str(TAX),
            "--out",
            str(tmp_path / "o.csv"),
            "--allowed-status",
            "direct,approximate",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_refuses_when_no_train_split(tmp_path):
    _make(tmp_path)
    r = _run(tmp_path)
    assert r.returncode != 0 and "train" in r.stderr


def test_derived_split_keeps_outfits_together_and_is_reproducible(tmp_path):
    _make(tmp_path)
    assert _run(tmp_path, "--derive-split").returncode == 0
    a = pd.read_csv(tmp_path / "o.csv")
    assert (a.groupby("group").split.nunique() == 1).all()
    assert a.split_source.str.contains("PENDING_TEAM_APPROVAL").all()
    _run(tmp_path, "--derive-split")
    b = pd.read_csv(tmp_path / "o.csv")
    assert a.equals(b)


def test_model_head_and_unfreeze(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.wardiq.clothing.resnet_category import build_model, models

    original = models.resnet50
    monkeypatch.setattr(models, "resnet50", lambda **kwargs: original(weights=None))

    m = build_model(5, [])
    assert sum(p.requires_grad for p in m.parameters()) == 2  # fc weight + bias only
    m2 = build_model(5, ["layer4"])
    assert sum(p.numel() for p in m2.parameters() if p.requires_grad) > sum(
        p.numel() for p in m.parameters() if p.requires_grad
    )


def test_fashion_mnist_uses_omar_split_unchanged(tmp_path):
    tax = json.loads(TAX.read_text(encoding="utf-8"))
    assert "fashion_mnist" in tax["sources"]
    man, spl = [], []
    for sp, n in (("train", 30), ("test", 8)):
        for i in range(n):
            iid = f"fashion_mnist_{sp}_{i:06d}"
            man.append(
                {
                    "item_id": iid,
                    "source_dataset": "Fashion-MNIST",
                    "source_split": sp,
                    "source_index": str(i),
                    "category_ids": str(i % 10),
                    "original_image_reference": "",
                }
            )
            spl.append(
                {
                    "item_id": iid,
                    "target_split": "validation" if (sp == "train" and i % 5 == 0) else sp,
                    "group_id": "",
                }
            )
    pd.DataFrame(man).to_csv(tmp_path / "m.csv", index=False)
    pd.DataFrame(spl).to_csv(tmp_path / "s.csv", index=False)
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_classification_manifest.py"),
            "--manifest",
            str(tmp_path / "m.csv"),
            "--splits",
            str(tmp_path / "s.csv"),
            "--taxonomy",
            str(TAX),
            "--out",
            str(tmp_path / "o.csv"),
            "--datasets",
            "Fashion-MNIST",
            "--label-space",
            "source_label",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    out = pd.read_csv(tmp_path / "o.csv")
    assert out.image_ref.str.startswith("fashion_mnist://").all()
    sp = out.merge(pd.DataFrame(spl), on="item_id")
    assert (sp.split == sp.target_split).all()  # Omar's assignment untouched
    assert out.split_source.eq("m1_split_manifest").all() and out.label.nunique() == 10


def test_checkpoint_reload_does_not_download_pretrained_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.wardiq.clothing import resnet_category as resnet

    model = resnet.build_model(2, [], pretrained=False)
    checkpoint = tmp_path / "synthetic_checkpoint.pt"
    torch.save(
        {"state_dict": model.state_dict(), "classes": ["0", "1"], "config": {"image_size": 32}},
        checkpoint,
    )
    original = resnet.models.resnet50

    def offline_resnet(**kwargs):
        assert kwargs["weights"] is None, "Inference must use the supplied checkpoint only"
        return original(**kwargs)

    monkeypatch.setattr(resnet.models, "resnet50", offline_resnet)
    reloaded, classes, size = resnet.load_for_inference(str(checkpoint))
    assert classes == ["0", "1"] and size == 32
    assert all(
        torch.equal(value, reloaded.state_dict()[name])
        for name, value in model.state_dict().items()
    )


def test_reload_check_handles_numeric_labels_and_rejects_mismatch(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.wardiq.clothing import resnet_category as resnet

    pd.DataFrame([{"item_id": "001", "pred": "0"}]).to_csv(
        tmp_path / "val_predictions.csv", index=False
    )
    frame = pd.DataFrame([{"item_id": "001", "image_ref": "synthetic"}])
    monkeypatch.setattr(resnet, "load_for_inference", lambda *args: (None, ["0", "1"], 32))
    monkeypatch.setattr(resnet, "open_image", lambda *args: None)
    monkeypatch.setattr(
        resnet, "predict_item", lambda *args: {"category": {"common_category": "0"}}
    )
    resnet.reload_check(tmp_path, {"image_root": str(tmp_path)}, frame, ["0", "1"], "cpu")
    assert json.loads((tmp_path / "reload_check.json").read_text()) == {"checked": 1, "matching": 1}
    monkeypatch.setattr(
        resnet, "predict_item", lambda *args: {"category": {"common_category": "1"}}
    )
    with pytest.raises(ValueError, match="differ"):
        resnet.reload_check(tmp_path, {"image_root": str(tmp_path)}, frame, ["0", "1"], "cpu")
