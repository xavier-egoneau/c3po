import struct

from llm_runtime.gguf import read_metadata, model_shape

# Types GGUF utilisés ici
_U32, _STR, _ARR = 4, 8, 9


def _str(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def _kv_str(key: bytes, val: bytes) -> bytes:
    return _str(key) + struct.pack("<I", _STR) + _str(val)


def _kv_u32(key: bytes, val: int) -> bytes:
    return _str(key) + struct.pack("<I", _U32) + struct.pack("<I", val)


def _kv_str_array(key: bytes, items: list[bytes]) -> bytes:
    body = _str(key) + struct.pack("<I", _ARR)
    body += struct.pack("<I", _STR) + struct.pack("<Q", len(items))
    for it in items:
        body += _str(it)
    return body


def _build_gguf(kvs: list[bytes], version: int = 3) -> bytes:
    header = b"GGUF" + struct.pack("<I", version)
    header += struct.pack("<Q", 0)            # tensor_count
    header += struct.pack("<Q", len(kvs))     # kv_count
    return header + b"".join(kvs)


def test_read_metadata_extracts_wanted_keys(tmp_path):
    data = _build_gguf([
        _kv_str(b"general.architecture", b"qwen2"),
        # gros tableau de strings à sauter (simule le vocabulaire tokenizer)
        _kv_str_array(b"tokenizer.ggml.tokens", [b"a", b"bb", b"ccc"] * 100),
        _kv_u32(b"qwen2.block_count", 28),
        _kv_u32(b"qwen2.context_length", 32768),
        _kv_u32(b"qwen2.embedding_length", 3584),
        _kv_u32(b"qwen2.attention.head_count", 28),
        _kv_u32(b"qwen2.attention.head_count_kv", 4),
    ])
    path = tmp_path / "tiny.gguf"
    path.write_bytes(data)

    meta = read_metadata(path)
    assert meta["general.architecture"] == "qwen2"
    assert meta["qwen2.block_count"] == 28
    assert meta["qwen2.context_length"] == 32768
    # head_count et head_count_kv ne doivent pas se confondre
    assert meta["qwen2.attention.head_count"] == 28
    assert meta["qwen2.attention.head_count_kv"] == 4
    # le tableau sauté n'est pas matérialisé
    assert "tokenizer.ggml.tokens" not in meta


def test_model_shape_maps_dimensions(tmp_path):
    data = _build_gguf([
        _kv_str(b"general.architecture", b"llama"),
        _kv_u32(b"llama.block_count", 32),
        _kv_u32(b"llama.context_length", 8192),
        _kv_u32(b"llama.embedding_length", 4096),
        _kv_u32(b"llama.attention.head_count", 32),
        _kv_u32(b"llama.attention.head_count_kv", 8),
    ])
    path = tmp_path / "m.gguf"
    path.write_bytes(data)

    shape = model_shape(path)
    assert shape == {
        "arch": "llama", "n_layers": 32, "n_ctx_train": 8192,
        "n_embd": 4096, "n_heads": 32, "n_kv_heads": 8,
    }


def test_model_shape_returns_none_on_non_gguf(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"not a gguf file at all")

    shape = model_shape(path)
    assert shape == {
        "arch": None, "n_layers": None, "n_ctx_train": None,
        "n_embd": None, "n_heads": None, "n_kv_heads": None,
    }
