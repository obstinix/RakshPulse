import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rakshpulse.models.cnn import BPNet, CNNConfig, CNNRegressor, n_params  # noqa: E402


def test_bpnet_shapes_and_size():
    m = BPNet(in_ch=2)
    out = m(torch.zeros(3, 2, 1000))
    assert out.shape == (3, 2)
    assert n_params(m) < 50_000  # microcontroller-scale budget


def test_cnn_regressor_fit_predict_smoke():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 1, 1000)).astype(np.float32)
    y = np.column_stack([rng.normal(120, 10, 40), rng.normal(75, 8, 40)])
    groups = np.repeat(np.arange(10), 4)
    m = CNNRegressor(1, CNNConfig(epochs=2, patience=1)).fit(X, y, groups)
    p = m.predict(X)
    assert p.shape == (40, 2) and np.isfinite(p).all()
