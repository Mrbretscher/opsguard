from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_installs_gui_app_and_exposes_streamlit() -> None:
    dockerfile = PROJECT_ROOT / "Dockerfile"

    content = dockerfile.read_text(encoding="utf-8")

    assert 'python -m pip install ".[app]"' in content
    assert "EXPOSE 8501" in content
    assert "streamlit" in content
    assert "_stcore/health" in content


def test_dockerignore_excludes_generated_artifacts() -> None:
    dockerignore = PROJECT_ROOT / ".dockerignore"

    entries = {
        line.strip()
        for line in dockerignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert "data/**" in entries
    assert "reports/" in entries
    assert "models/" in entries
    assert "outputs/" in entries
    assert ".venv/" in entries


def test_docker_smoke_script_checks_streamlit_health() -> None:
    script = PROJECT_ROOT / "scripts" / "docker_smoke_test.ps1"

    content = script.read_text(encoding="utf-8")

    assert "docker build" in content
    assert "docker run" in content
    assert "_stcore/health" in content
    assert "docker stop" in content
