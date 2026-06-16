from llm_runtime.stats import BenchmarkStats, ModelStats, format_stats


def test_format_stats_shows_general_and_code_benchmarks():
    stats = ModelStats(
        name="model",
        path="model.gguf",
        size_gb=4.0,
        architecture="qwen2",
        quantization="Q4_K_M",
        n_params_b=None,
        n_ctx_train=32768,
        n_embd=3584,
        n_layers=28,
        n_vocab=152064,
        backend="cuda",
        device="gpu",
        n_gpu_layers=-1,
        n_threads=8,
        n_ctx=4096,
        flash_attn=True,
        kv_type="f16",
        speculative=True,
        repeat_penalty=1.1,
        load_time_s=1.2,
        ttft_s=0.02,
        gen_tps=80.0,
        gen_tokens=100,
        vram_used_mb=4800,
        benchmarks=[
            BenchmarkStats("general", "question ouverte", 0.02, 80.0, 100),
            BenchmarkStats("code", "réécriture code/édition", 0.03, 130.0, 120),
        ],
    )

    rendered = format_stats(stats)

    assert "general (question ouverte, 100 tokens)" in rendered
    assert "code (réécriture code/édition, 120 tokens)" in rendered
    assert "80.0 tok/s" in rendered
    assert "130.0 tok/s" in rendered
