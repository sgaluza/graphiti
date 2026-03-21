"""
LLM client that delegates to the Claude CLI (`claude -p`).

Uses `--json-schema` for structured output and `--model` for model selection.
Requires the `claude` CLI to be installed and authenticated.
"""

import asyncio
import json
import logging
import typing

from pydantic import BaseModel

from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError
from ..prompts.models import Message

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'claude-sonnet-4-5-latest'
DEFAULT_SMALL_MODEL = 'claude-haiku-4-5-latest'


class ClaudeCliClient(LLMClient):
    """
    LLM client that calls the Claude CLI subprocess instead of the API directly.

    Useful when you want to delegate model selection and auth to the CLI,
    or when running in an environment where the CLI is already configured.

    Structured output is handled natively via `--json-schema`, so the model
    returns a validated JSON object in `structured_output` field.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        claude_bin: str = 'claude',
    ):
        """
        Args:
            config: LLM config. model/small_model map to --model flag.
            cache: Whether to enable response caching.
            claude_bin: Path to claude CLI binary. Defaults to 'claude' (from PATH).
        """
        if config is None:
            config = LLMConfig()
        if config.model is None:
            config.model = DEFAULT_MODEL
        if config.small_model is None:
            config.small_model = DEFAULT_SMALL_MODEL

        super().__init__(config, cache)
        self.claude_bin = claude_bin

    def _build_prompt(self, messages: list[Message]) -> str:
        """Concatenate messages into a single prompt string for the CLI."""
        parts = []
        for m in messages:
            if m.role == 'system':
                parts.append(f'<system>\n{m.content}\n</system>')
            elif m.role == 'user':
                parts.append(m.content)
            elif m.role == 'assistant':
                parts.append(f'<assistant>\n{m.content}\n</assistant>')
        return '\n\n'.join(parts)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        model = self.model if model_size == ModelSize.medium else self.small_model
        prompt = self._build_prompt(messages)

        cmd = [
            self.claude_bin,
            '-p', prompt,
            '--output-format', 'json',
            '--model', model,
        ]

        if response_model is not None:
            schema = json.dumps(response_model.model_json_schema())
            cmd += ['--json-schema', schema]

        logger.debug(f'ClaudeCliClient: running {self.claude_bin} with model={model}')

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip()
            if 'rate limit' in err.lower():
                raise RateLimitError(err)
            raise RuntimeError(f'Claude CLI exited {proc.returncode}: {err}')

        data = json.loads(stdout.decode())

        if data.get('is_error'):
            raise RuntimeError(f'Claude CLI error: {data.get("result", "unknown error")}')

        # Structured output is in `structured_output` when --json-schema is used,
        # otherwise fall back to parsing `result` as JSON text.
        if 'structured_output' in data and data['structured_output']:
            return data['structured_output']

        result_text = data.get('result', '')
        if not result_text:
            return {}

        # Try to parse result as JSON (model may wrap it in ```json ... ```)
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            # Strip markdown code fences and retry
            stripped = result_text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning('ClaudeCliClient: could not parse response as JSON, returning raw text')
                return {'result': result_text}
