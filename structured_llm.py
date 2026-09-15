"""Structured LLM extraction utility, ported into cat-graphrag.my.

Import-safe port of the ``StructuredLLM`` class from the MyCAT reference on branch
``feat/structured-output`` (commit ``0594ba167eb8a1c008f51254c3d8672b6d35c143``).

The Cat plugin loader host-imports every ``.py`` under the plugin folder
(``cat.plugins.<id>.<file>``) at activation, so this module keeps its top level to
the Python standard library only. All third-party imports (``langchain_core``,
``jsonschema``, pydantic, ``cat.utils``) are lazy — they happen inside the function
bodies that use them. ``from __future__ import annotations`` keeps the ``BaseModel``
type hint below unevaluated at import time.

API contract (see .omo/plans/structured-json-agentic-workflow.md):
  - ``StructuredLLM.run(prompt, llm, output_model, system_prompt=None)``: calls the
    supplied LLM for structured JSON. Native ``llm.with_structured_output`` when
    supported, else a JSON-schema prompt + parse/validate fallback.
  - ``StructuredLLMResult``: ``.text`` (raw JSON str) and ``.model`` (BaseModel | dict | None).
  - ``StructuredLLMError``: controlled error carrying ``.raw_text``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class StructuredLLMError(Exception):
    """Raised when the LLM output cannot be parsed/validated into the output model."""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class StructuredLLMResult:
    text: str
    model: Any | None


class StructuredLLM:
    async def run(
        self,
        prompt: str,
        llm: Any,
        output_model: dict[str, Any] | type[BaseModel],  # lazy annotation (future import)
        system_prompt: str | None = None,
    ) -> StructuredLLMResult:
        if hasattr(llm, "with_structured_output"):
            return await self._run_native(prompt, llm, output_model)
        return await self._run_fallback(prompt, llm, output_model, system_prompt)

    @staticmethod
    def _is_pydantic(obj) -> bool:
        return hasattr(obj, "model_json_schema")

    async def _run_native(self, prompt: str, llm: Any, output_model: Any) -> StructuredLLMResult:
        # lazy import (import-safe plugin rule)
        from langchain_core.runnables import RunnableConfig

        try:
            bounded = llm.with_structured_output(output_model)
            res = await bounded.ainvoke(prompt, config=RunnableConfig())

            if hasattr(res, "model_dump_json"):
                text = res.model_dump_json()
                model = res
            elif isinstance(res, dict):
                text = json.dumps(res)
                if self._is_pydantic(output_model):
                    model = output_model(**res)
                else:
                    model = res
            else:
                text = str(res)
                model = None
                # lazy import (import-safe plugin rule): top level stays stdlib-only
                from cat.log import log

                log.warning(
                    f"[GraphRAG] Structured LLM returned non-structured result: {str(res)[:200]}"
                )
        except Exception as e:
            raw = getattr(e, "raw_text", "")
            raise StructuredLLMError(f"Native structured output failed: {e}", raw)

        return StructuredLLMResult(text=text, model=model)

    async def _run_fallback(
        self, prompt: str, llm: Any, output_model: Any, system_prompt: str | None
    ) -> StructuredLLMResult:
        # lazy imports (import-safe plugin rule)
        from langchain_core.runnables import RunnableConfig

        try:
            import jsonschema  # gated: skip schema validation where unavailable
        except ImportError:
            jsonschema = None
        try:
            from cat.utils import parse_json
        except ImportError:
            # cat.utils not importable (e.g. standalone QA harness): fall back to plain json.
            parse_json = None

        if self._is_pydantic(output_model):
            schema = output_model.model_json_schema()
        else:
            schema = output_model
        full_prompt = self._build_json_prompt(schema, prompt, system_prompt)

        raw = await llm.ainvoke(full_prompt, config=RunnableConfig())
        text = getattr(raw, "content", str(raw))

        try:
            if self._is_pydantic(output_model):
                if parse_json is not None:
                    model = parse_json(text, output_model)
                else:
                    model = output_model(**json.loads(text))
            else:
                model = json.loads(text)
                if jsonschema is not None:
                    jsonschema.validate(model, output_model)
        except Exception as e:
            if jsonschema is not None and isinstance(e, jsonschema.exceptions.ValidationError):
                raise StructuredLLMError(
                    f"Failed to validate LLM output against JSON schema: {e}", text
                )
            raise StructuredLLMError(f"Failed to parse LLM output as structured JSON: {e}", text)

        return StructuredLLMResult(text=text, model=model)

    @staticmethod
    def _build_json_prompt(schema: dict, prompt: str, system_prompt: str | None) -> str:
        schema_json = json.dumps(schema)
        if system_prompt:
            return (
                f"{system_prompt}\n\n"
                f"Return ONLY a single valid JSON object conforming to this JSON schema:\n"
                f"```json\n{schema_json}\n```\n\n"
                f"Input:\n{prompt}"
            )
        return (
            "Return ONLY a single valid JSON object conforming to this JSON schema:\n"
            f"```json\n{schema_json}\n```\n\n"
            f"Input:\n{prompt}"
        )