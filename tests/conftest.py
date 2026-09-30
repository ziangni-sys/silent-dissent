from pathlib import Path

import pytest

from silent_dissent.data import load_items

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def items():
    return load_items(str(FIXTURES / "items.jsonl"))
