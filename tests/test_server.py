from llm_runtime.server import ChatMessage, ChatCompletionRequest


def test_chat_message_model():
    msg = ChatMessage(role="user", content="bonjour")
    assert msg.role == "user"
    assert msg.content == "bonjour"


def test_chat_completion_request_defaults():
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "salut"}])

    assert request.model is None
    assert request.max_tokens == 512
    assert request.temperature == 0.7
    assert request.stream is False
    assert request.messages[0].role == "user"


def test_chat_completion_request_with_model():
    request = ChatCompletionRequest(
        model="qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": "salut"}],
        stream=True,
    )

    assert request.model == "qwen2.5-7b-instruct"
    assert request.stream is True
