import pytest
from mlp.fcn_runner import FCNRunner
from unittest.mock import MagicMock
import numpy as np
from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays

@pytest.fixture
def fcn_runner():
    config = MagicMock()
    params = MagicMock()
    fcn_runner = FCNRunner(config, params)
    return fcn_runner

@given (
   input = arrays(np.int32, shape=st.integers(1, 25), elements=st.integers(0, 1)),
   batch_size = st.integers(0, 25)
)
def test_split_to_batches(fcn_runner, input, batch_size):
    assume(batch_size > 0)
    c = fcn_runner.split_to_batches(input, batch_size)
    if len(c) > 1:
        k = set([k.shape[0] for k in c[:-1]])
        assert len(k) == 1
    result = np.concatenate(c)
    np.testing.assert_array_equal(result, input)

