import json
import os
import pytest
from importlib import import_module, reload


@pytest.fixture(scope="module")
def app_module():
    # Load app.py against the Azure Search test settings
    os.environ["DOTENV_PATH"] = os.path.join(
        os.path.dirname(__file__), "dotenv_data", "dotenv_with_azure_search_success"
    )
    reload(import_module("backend.settings"))
    reload(import_module("backend.filter_compiler"))
    return reload(import_module("app"))


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _fake_client(captured, rows):
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, params=None, headers=None, json=None):
            captured.update({"url": url, "params": params, "headers": headers, "body": json})
            return _FakeResponse({"value": rows})

    return _FakeAsyncClient


@pytest.mark.asyncio
async def test_search_docs_hybrid_semantic_applies_strictness(app_module, monkeypatch):
    ds = app_module.app_settings.datasource
    monkeypatch.setattr(ds, "query_type", "vector_semantic_hybrid")
    monkeypatch.setattr(ds, "semantic_search_config", "my-semantic")
    monkeypatch.setattr(ds, "strictness", 3)

    rows = [
        {"@search.rerankerScore": 2.5, "content1": "a", "content2": "b", "title": "T1", "url": "u1", "filepath": "f1"},
        {"@search.rerankerScore": 0.4, "content1": "weak", "title": "T2", "url": "u2", "filepath": "f2"},
    ]
    captured = {}
    monkeypatch.setattr(app_module.httpx, "AsyncClient", _fake_client(captured, rows))

    citations = await app_module._search_docs("bridge projects", "(client eq 'X')")

    body = captured["body"]
    assert captured["url"].endswith("/indexes/search_index/docs/search")
    assert body["search"] == "bridge projects"
    assert body["filter"] == "(client eq 'X')"
    assert body["queryType"] == "semantic"
    assert body["semanticConfiguration"] == "my-semantic"
    assert body["vectorQueries"][0]["kind"] == "text"
    assert body["vectorQueries"][0]["fields"] == "vector1"
    assert body["top"] == 20
    assert "api-key" in captured["headers"]

    # weak result dropped by strictness 3
    assert citations == [
        {"content": "a\nb", "title": "T1", "url": "u1", "filepath": "f1", "chunk_id": "0"}
    ]


@pytest.mark.asyncio
async def test_search_docs_simple_query(app_module, monkeypatch):
    ds = app_module.app_settings.datasource
    monkeypatch.setattr(ds, "query_type", "simple")
    captured = {}
    monkeypatch.setattr(app_module.httpx, "AsyncClient", _fake_client(captured, []))

    citations = await app_module._search_docs("roundabouts", None)

    body = captured["body"]
    assert citations == []
    assert body["top"] == 5
    assert "filter" not in body
    assert "vectorQueries" not in body
    assert "queryType" not in body


def test_grounded_system_message(app_module):
    msg = app_module._build_grounded_system_message(
        [{"content": "Bridge over river", "title": "Proposal A"}]
    )
    assert msg.startswith(app_module.OYD_ROLE_INFORMATION)
    assert "[doc1]\nTitle: Proposal A\nContent: Bridge over river" in msg
    assert app_module.OYD_NOT_IN_DATA_RESPONSE in msg

    assert "No documents were retrieved." in app_module._build_grounded_system_message([])


def test_prepare_model_args_has_no_data_sources(app_module):
    app_module.MS_DEFENDER_ENABLED = False
    args = app_module.prepare_model_args(
        {"messages": [{"role": "user", "content": "hi"}]}, {}, citations=[]
    )
    assert "extra_body" not in args
    assert args["messages"][0]["role"] == "system"
    assert args["messages"][1] == {"role": "user", "content": "hi"}


def test_format_history_strips_citations_and_limits():
    from backend.filter_compiler import _format_history, HISTORY_MAX_MESSAGES

    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    history.append({"role": "assistant", "content": "See proposal [doc1][doc2]."})
    history.append({"role": "tool", "content": "{}"})

    lines = _format_history(history).splitlines()
    assert len(lines) == HISTORY_MAX_MESSAGES - 1  # tool message skipped
    assert lines[-1] == "assistant: See proposal ."
