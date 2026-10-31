import json
import pytest
from types import SimpleNamespace
from backend.utils import (
    format_as_ndjson,
    format_context_message,
    format_non_streaming_response,
    parse_multi_columns,
)


def test_format_context_message():
    context = {"citations": [{"content": "c"}], "intent": json.dumps(["q"])}
    result = format_context_message(context, {"title": "t"}, "apim-1", "gpt")

    messages = result["choices"][0]["messages"]
    assert messages == [{"role": "tool", "content": json.dumps(context)}]
    assert result["id"]
    assert result["history_metadata"] == {"title": "t"}
    assert result["apim-request-id"] == "apim-1"
    # frontend parses intent as a JSON string inside the tool content
    assert json.loads(json.loads(messages[0]["content"])["intent"]) == ["q"]


def test_format_non_streaming_response_with_context():
    completion = SimpleNamespace(
        id="1", model="gpt", created=0, object="chat.completion",
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer [doc1]"))],
    )
    context = {"citations": [], "intent": "[]"}
    result = format_non_streaming_response(completion, {}, "apim-1", context)

    assert result["choices"][0]["messages"] == [
        {"role": "tool", "content": json.dumps(context)},
        {"role": "assistant", "content": "answer [doc1]"},
    ]


@pytest.mark.asyncio
async def test_format_as_ndjson():
    async def dummy_generator():
        yield {"message": "test message\n"}

    async for event in format_as_ndjson(dummy_generator()):
        assert event == '{"message": "test message\\n"}\n'


@pytest.mark.asyncio
async def test_format_as_ndjson_exception():
    async def dummy_generator():
        raise Exception("test exception")
        yield {"message": "test message\n"}
    
    async for event in format_as_ndjson(dummy_generator()):
        assert event == '{"error": "test exception"}'

def test_parse_multi_columns():
    test_pipes = "col1|col2|col3"
    test_commas = "col1,col2,col3"
    test_single = "col1"
    assert parse_multi_columns(test_pipes) == ["col1", "col2", "col3"]
    assert parse_multi_columns(test_commas) == ["col1", "col2", "col3"]
    assert parse_multi_columns(test_single) == ["col1"]
