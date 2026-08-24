from __future__ import annotations

import pytest

from app.agent.providers import _parse_wait_seconds


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2.347s", pytest.approx(2.347)),
        ("15m50.4s", pytest.approx(15 * 60 + 50.4)),
        ("5", pytest.approx(5.0)),
        (None, None),
        ("not-a-duration", None),
    ],
)
def test_parse_wait_seconds(value, expected):
    assert _parse_wait_seconds(value) == expected
