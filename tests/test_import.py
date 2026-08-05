def test_package_import() -> None:
    import opsguard

    assert opsguard.__version__ == "0.1.0"
