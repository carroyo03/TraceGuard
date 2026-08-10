import traceguard


def test_package_convenience_export_is_available_lazily() -> None:
    assert callable(traceguard.run_protected_agent)
