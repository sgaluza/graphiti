# Running tests: pytest -xvs tests/llm_client/test_claude_cli_client.py

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.claude_cli_client import ClaudeCliClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message


class PersonModel(BaseModel):
    name: str
    age: int


def make_cli_response(
    result: str = '',
    structured_output: dict | None = None,
    is_error: bool = False,
) -> bytes:
    """Build a JSON response mimicking `claude -p --output-format json`."""
    data: dict = {'type': 'result', 'is_error': is_error, 'result': result}
    if structured_output is not None:
        data['structured_output'] = structured_output
    return json.dumps(data).encode()


def make_process_mock(stdout: bytes, returncode: int = 0, stderr: bytes = b'') -> AsyncMock:
    """Create an asyncio.Process mock."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


@pytest.fixture
def client() -> ClaudeCliClient:
    config = LLMConfig(model='claude-opus-4-6', small_model='claude-haiku-4-5-latest')
    return ClaudeCliClient(config=config, cache=False)


class TestClaudeCliClientInit:
    def test_default_models_set_when_none(self):
        c = ClaudeCliClient()
        assert c.model is not None
        assert c.small_model is not None

    def test_custom_claude_bin(self):
        c = ClaudeCliClient(claude_bin='/usr/local/bin/claude')
        assert c.claude_bin == '/usr/local/bin/claude'

    def test_config_preserved(self, client):
        assert client.model == 'claude-opus-4-6'
        assert client.small_model == 'claude-haiku-4-5-latest'


class TestBuildPrompt:
    def test_user_message_only(self, client):
        messages = [Message(role='user', content='Hello')]
        assert client._build_prompt(messages) == 'Hello'

    def test_system_message_wrapped(self, client):
        messages = [Message(role='system', content='Be helpful')]
        prompt = client._build_prompt(messages)
        assert '<system>' in prompt
        assert 'Be helpful' in prompt

    def test_assistant_message_wrapped(self, client):
        messages = [Message(role='assistant', content='Sure')]
        prompt = client._build_prompt(messages)
        assert '<assistant>' in prompt

    def test_messages_joined_with_double_newline(self, client):
        messages = [
            Message(role='system', content='sys'),
            Message(role='user', content='user'),
        ]
        prompt = client._build_prompt(messages)
        assert '\n\n' in prompt


class TestGenerateResponse:
    @pytest.mark.asyncio
    async def test_returns_structured_output_when_response_model_given(self, client):
        stdout = make_cli_response(structured_output={'name': 'Alice', 'age': 30})

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)):
            result = await client._generate_response(
                messages=[Message(role='user', content='Extract person')],
                response_model=PersonModel,
            )

        assert result == {'name': 'Alice', 'age': 30}

    @pytest.mark.asyncio
    async def test_passes_json_schema_flag_when_response_model_given(self, client):
        stdout = make_cli_response(structured_output={'name': 'Bob', 'age': 25})

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)) as mock_exec:
            await client._generate_response(
                messages=[Message(role='user', content='Extract')],
                response_model=PersonModel,
            )

        call_args = mock_exec.call_args[0]
        assert '--json-schema' in call_args
        schema_idx = list(call_args).index('--json-schema')
        schema = json.loads(call_args[schema_idx + 1])
        assert 'properties' in schema
        assert 'name' in schema['properties']

    @pytest.mark.asyncio
    async def test_returns_parsed_result_text_when_no_response_model(self, client):
        stdout = make_cli_response(result='{"key": "value"}')

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)):
            result = await client._generate_response(
                messages=[Message(role='user', content='Give me JSON')],
            )

        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_strips_markdown_fences_from_result(self, client):
        raw = '```json\n{"key": "value"}\n```'
        stdout = make_cli_response(result=raw)

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)):
            result = await client._generate_response(
                messages=[Message(role='user', content='Give JSON')],
            )

        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_returns_raw_text_dict_when_unparseable(self, client):
        stdout = make_cli_response(result='just plain text')

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)):
            result = await client._generate_response(
                messages=[Message(role='user', content='Hello')],
            )

        assert 'result' in result
        assert result['result'] == 'just plain text'

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_result_is_empty(self, client):
        stdout = make_cli_response(result='')

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)):
            result = await client._generate_response(
                messages=[Message(role='user', content='Hello')],
            )

        assert result == {}

    @pytest.mark.asyncio
    async def test_raises_rate_limit_error_on_rate_limit_stderr(self, client):
        with patch(
            'asyncio.create_subprocess_exec',
            return_value=make_process_mock(b'', returncode=1, stderr=b'rate limit exceeded'),
        ):
            with pytest.raises(RateLimitError):
                await client._generate_response(
                    messages=[Message(role='user', content='Hello')],
                )

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_nonzero_exit(self, client):
        with patch(
            'asyncio.create_subprocess_exec',
            return_value=make_process_mock(b'', returncode=1, stderr=b'something went wrong'),
        ):
            with pytest.raises(RuntimeError, match='Claude CLI exited 1'):
                await client._generate_response(
                    messages=[Message(role='user', content='Hello')],
                )

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_is_error_true(self, client):
        stdout = make_cli_response(result='model refused', is_error=True)

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)):
            with pytest.raises(RuntimeError, match='Claude CLI error'):
                await client._generate_response(
                    messages=[Message(role='user', content='Hello')],
                )

    @pytest.mark.asyncio
    async def test_uses_small_model_for_small_model_size(self, client):
        stdout = make_cli_response(result='{}')

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)) as mock_exec:
            await client._generate_response(
                messages=[Message(role='user', content='Hello')],
                model_size=ModelSize.small,
            )

        call_args = mock_exec.call_args[0]
        assert '--model' in call_args
        model_idx = list(call_args).index('--model')
        assert call_args[model_idx + 1] == client.small_model

    @pytest.mark.asyncio
    async def test_uses_medium_model_by_default(self, client):
        stdout = make_cli_response(result='{}')

        with patch('asyncio.create_subprocess_exec', return_value=make_process_mock(stdout)) as mock_exec:
            await client._generate_response(
                messages=[Message(role='user', content='Hello')],
            )

        call_args = mock_exec.call_args[0]
        model_idx = list(call_args).index('--model')
        assert call_args[model_idx + 1] == client.model
