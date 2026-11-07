from pathlib import Path


def test_tif1_is_isolated_from_ml_api_requirements():
    root = Path(__file__).resolve().parents[1]
    runtime_requirements = (root / "requirements.txt").read_text().lower()
    tif1_requirements = (root / "requirements-tif1.txt").read_text().lower()

    assert "pandas>=2.2,<3.0" in runtime_requirements
    assert not any(line.strip().startswith("tif1") for line in runtime_requirements.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    assert "tif1==0.9.0" in tif1_requirements


def test_windows_setup_documents_separate_tif1_environment():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text().lower()

    assert ".venv-tif1" in readme
    assert "requirements-tif1.txt" in readme
    assert "install tif1 only in a separate collector environment" in readme
