import pytest

from openpi.models import tokenizer as _tokenizer

pytestmark = [pytest.mark.network, pytest.mark.data]


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)
