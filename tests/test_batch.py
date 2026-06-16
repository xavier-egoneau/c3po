from pathlib import Path
from llm_runtime.batch import optimal_jobs, tasks_from_files, tasks_from_prompts


def _make_gguf(tmp_path: Path, size_mb: int) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0" * (size_mb * 1024 * 1024))
    return path


def test_tasks_from_files_injects_content(tmp_path):
    f1 = tmp_path / "doc1.txt"
    f1.write_text("contenu 1")
    f2 = tmp_path / "doc2.txt"
    f2.write_text("contenu 2")

    tasks = tasks_from_files([f1, f2], prompt_template="Résume : {content}")

    assert len(tasks) == 2
    assert tasks[0].prompt == "Résume : contenu 1"
    assert tasks[1].prompt == "Résume : contenu 2"
    assert tasks[0].mode == "file"
    assert tasks[0].source == str(f1)


def test_tasks_from_prompts():
    tasks = tasks_from_prompts(["a", "b", "c"])

    assert [t.prompt for t in tasks] == ["a", "b", "c"]
    assert all(t.mode == "prompt" for t in tasks)


def test_optimal_jobs_is_one_on_metal(tmp_path):
    model_path = _make_gguf(tmp_path, size_mb=100)  # ~0.098 Go

    jobs = optimal_jobs(model_path, n_ctx=2048)

    assert jobs == 1


def test_optimal_jobs_is_one_on_cuda(tmp_path):
    # GPU Nvidia : VRAM dédiée + sérialisation → une seule instance, peu importe la taille.
    model_path = _make_gguf(tmp_path, size_mb=100)

    jobs = optimal_jobs(model_path, n_ctx=2048)

    assert jobs == 1


def test_optimal_jobs_stays_one_with_larger_ctx(tmp_path):
    model_path = _make_gguf(tmp_path, size_mb=2000)  # ~1.95 Go

    jobs_small_ctx = optimal_jobs(model_path, n_ctx=2048)
    jobs_large_ctx = optimal_jobs(model_path, n_ctx=16384)

    assert jobs_small_ctx == jobs_large_ctx == 1
