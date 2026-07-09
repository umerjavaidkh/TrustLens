"""Model tests. Skipped automatically where torch/torchvision aren't installed
(e.g. CI without a GPU); they run on Colab/Kaggle."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from trustlens.models.efficientnet import (  # noqa: E402
    build_model, freeze_backbone, unfreeze_last_blocks, trainable_parameter_count,
)


def test_build_model_output_shape():
    model = build_model(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


def test_freeze_backbone_only_head_trainable():
    model = build_model(num_classes=2, pretrained=False)
    freeze_backbone(model)
    trn, tot = trainable_parameter_count(model)
    assert trn < tot
    # Every trainable param must live in the classifier head.
    head_ids = {id(p) for p in model.classifier.parameters()}
    for p in model.parameters():
        if p.requires_grad:
            assert id(p) in head_ids


def test_unfreeze_last_blocks_increases_trainable():
    model = build_model(num_classes=2, pretrained=False)
    freeze_backbone(model)
    frozen, _ = trainable_parameter_count(model)
    unfreeze_last_blocks(model, n_blocks=2)
    unfrozen, _ = trainable_parameter_count(model)
    assert unfrozen > frozen


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = build_model(num_classes=2, pretrained=False)
    freeze_backbone(model)
    x = torch.randn(4, 3, 64, 64)  # small: just checks the step wires up
    y = torch.tensor([0, 1, 0, 1])
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    l0 = crit(model(x), y)
    for _ in range(5):
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    assert float(loss) < float(l0)
