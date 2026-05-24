"""Registry edge cases."""

from __future__ import annotations

import pytest

from ner_detector.registry import clear_backend_cache, create_backend


def test_unknown_backend_raises() -> None:
    clear_backend_cache()
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("invalid")  # type: ignore[arg-type]
