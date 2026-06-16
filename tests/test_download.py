import pytest

from llm_runtime.download import (
    GGUFFile, parse_ref, group_by_quant, choose_quant, best_fitting_quant, has_mmproj,
    select_mmproj,
)


def test_parse_ref_with_quant():
    assert parse_ref("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M") == (
        "bartowski/Qwen2.5-7B-Instruct-GGUF", "Q4_K_M"
    )


def test_parse_ref_without_quant():
    assert parse_ref("bartowski/Foo-GGUF") == ("bartowski/Foo-GGUF", None)


def test_parse_ref_requires_explicit_repo():
    with pytest.raises(ValueError, match="repo Hugging Face complet"):
        parse_ref("qwen2.5-7b")


def _files():
    return [
        GGUFFile("Model-Q4_K_M.gguf", size=4 * 1024**3, sha256="a"),
        GGUFFile("Model-Q8_0.gguf", size=8 * 1024**3, sha256="b"),
        GGUFFile("README.md".replace(".md", ".gguf").replace("README", "Model-Q2_K"),
                 size=2 * 1024**3, sha256="c"),
    ]


def test_group_by_quant_extracts_quants():
    opts = group_by_quant(_files())
    assert set(opts) == {"Q4_K_M", "Q8_0", "Q2_K"}
    assert opts["Q8_0"].total_size == 8 * 1024**3


def test_group_by_quant_merges_shards():
    files = [
        GGUFFile("Big-Q4_K_M-00001-of-00002.gguf", size=5 * 1024**3, sha256="x"),
        GGUFFile("Big-Q4_K_M-00002-of-00002.gguf", size=5 * 1024**3, sha256="y"),
    ]
    opts = group_by_quant(files)
    assert set(opts) == {"Q4_K_M"}
    assert len(opts["Q4_K_M"].files) == 2
    assert opts["Q4_K_M"].total_size == 10 * 1024**3
    # ordre déterministe des shards
    assert [f.path for f in opts["Q4_K_M"].files] == [
        "Big-Q4_K_M-00001-of-00002.gguf", "Big-Q4_K_M-00002-of-00002.gguf"
    ]


def test_group_by_quant_does_not_sum_redundant_single_and_shards():
    # Même quant offerte en fichier unique ET en shards : ne pas additionner.
    files = [
        GGUFFile("M-Q4_K_M-00001-of-00002.gguf", size=4 * 1024**3, sha256="a"),
        GGUFFile("M-Q4_K_M-00002-of-00002.gguf", size=1 * 1024**3, sha256="b"),
        GGUFFile("M-Q4_K_M.gguf", size=5 * 1024**3, sha256="c"),
    ]
    opts = group_by_quant(files)
    assert set(opts) == {"Q4_K_M"}
    assert opts["Q4_K_M"].total_size == 5 * 1024**3  # shards (4+1), pas 4+1+5


def test_group_by_quant_keeps_largest_on_quant_collision():
    # Deux bases distinctes partageant une quant : on garde la plus grosse.
    files = [
        GGUFFile("Model-BF16.gguf", size=14 * 1024**3, sha256="a"),
        GGUFFile("Model-old-BF16.gguf", size=10 * 1024**3, sha256="b"),
    ]
    opts = group_by_quant(files)
    assert opts["BF16"].total_size == 14 * 1024**3


def test_group_by_quant_excludes_mmproj():
    # Repo multimodal : le mmproj ne doit pas apparaître comme une quant téléchargeable.
    files = [
        GGUFFile("Model-Q4_K_M.gguf", size=4 * 1024**3, sha256="a"),
        GGUFFile("mmproj-F16.gguf", size=1 * 1024**3, sha256="b"),
    ]
    opts = group_by_quant(files)
    assert set(opts) == {"Q4_K_M"}
    assert has_mmproj(files) is True


def test_has_mmproj_false_for_text_only():
    files = [GGUFFile("Model-Q4_K_M.gguf", size=4 * 1024**3, sha256="a")]
    assert has_mmproj(files) is False


def test_select_mmproj_prefers_bf16_over_f32():
    files = [
        GGUFFile("mmproj-F32.gguf", size=2 * 1024**3, sha256="a"),
        GGUFFile("mmproj-BF16.gguf", size=1 * 1024**3, sha256="b"),
        GGUFFile("Model-Q4_K_M.gguf", size=4 * 1024**3, sha256="c"),
    ]

    chosen = select_mmproj(files)

    assert chosen is not None
    assert chosen.path == "mmproj-BF16.gguf"


def test_choose_quant_picks_largest_that_fits():
    opts = group_by_quant(_files())
    # 6 Go dispo : Q4_K_M (4*1.15=4.6) tient, Q8_0 (8*1.15=9.2) non → Q4_K_M
    chosen = choose_quant(opts, available_gb=6.0)
    assert chosen.quant == "Q4_K_M"


def test_choose_quant_respects_explicit_request():
    opts = group_by_quant(_files())
    chosen = choose_quant(opts, available_gb=6.0, requested="q8_0")
    assert chosen.quant == "Q8_0"


def test_choose_quant_unknown_request_lists_available():
    opts = group_by_quant(_files())
    with pytest.raises(ValueError, match="Disponibles"):
        choose_quant(opts, available_gb=6.0, requested="Q5_K_M")


def test_choose_quant_falls_back_to_smallest(capsys):
    opts = group_by_quant(_files())
    # 1 Go dispo : rien ne tient → plus petit (Q2_K) + avertissement
    chosen = choose_quant(opts, available_gb=1.0)
    assert chosen.quant == "Q2_K"
    assert "Aucune quantization ne tient" in capsys.readouterr().err


def test_best_fitting_quant_when_something_fits():
    opts = group_by_quant(_files())
    best, fits = best_fitting_quant(opts, available_gb=6.0)
    assert (best.quant, fits) == ("Q4_K_M", True)


def test_best_fitting_quant_when_nothing_fits():
    opts = group_by_quant(_files())
    best, fits = best_fitting_quant(opts, available_gb=1.0)
    assert (best.quant, fits) == ("Q2_K", False)
