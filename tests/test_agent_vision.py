import base64

from agent.vision.gemma4 import (
    data_uri,
    find_mmproj,
    parse_jsonish,
    vision_messages,
)


def test_data_uri_encodes_image_file(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"abc")

    uri = data_uri(image)

    assert uri == "data:image/png;base64," + base64.b64encode(b"abc").decode("ascii")


def test_vision_messages_use_openai_style_image_part(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"abc")

    messages = vision_messages(image, "Décris")

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image_url"
    assert messages[0]["content"][1] == {"type": "text", "text": "Décris"}


def test_parse_jsonish_accepts_plain_and_fenced_json():
    assert parse_jsonish('{"caption": "ok"}') == {"caption": "ok"}
    assert parse_jsonish('```json\n{"caption": "ok"}\n```') == {"caption": "ok"}
    assert parse_jsonish("pas du json") is None


def test_find_mmproj_prefers_bf16(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    f32 = tmp_path / "mmproj-F32.gguf"
    bf16 = tmp_path / "mmproj-BF16.gguf"
    f32.write_bytes(b"f32")
    bf16.write_bytes(b"bf16")

    assert find_mmproj(model) == bf16
