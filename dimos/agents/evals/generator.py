# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eval generator module for creating fine-tuning datasets from DIMOS blueprints."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dimos.agents.evals.config import EvalGeneratorConfig
from dimos.agents.evals.prompts import (
    MULTI_TURN_SYSTEM_PROMPT,
    SINGLE_TURN_SYSTEM_PROMPT,
    build_multi_turn_prompt,
    build_single_turn_prompt,
)
from dimos.agents.evals.runner import ModelSpec
from dimos.agents.evals.schema_extractor import extract_skills_from_blueprint
from dimos.agents.spec import ToolSchemaList
from dimos.core.blueprints import ModuleBlueprintSet
from dimos.core.module import Module
from dimos.utils.logging_config import setup_logger

logger = setup_logger()
SYSTEM_MESSAGE = "You are a helpful robot assistant. Use the available tools to help the user."


def mk_tool_call(tc):
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": tc.get("name", ""), "arguments": json.dumps(tc.get("arguments", {}))},
    }


class EvalGenerator(Module):
    """Module for generating fine-tuning evaluation datasets from blueprints.

    Example:
        ```python
        from dimos.agents.evals import EvalGenerator, ModelSpec
        from dimos.agents.spec import Model, Provider
        from dimos.robot.all_blueprints import get_blueprint_by_name

        blueprint = get_blueprint_by_name('unitree-go2-agentic')
        gen = EvalGenerator(num_evals=100)
        model_spec = ModelSpec(model=Model.GPT_4O.value, provider=Provider.OPENAI)
        json_path = gen.generate_from_blueprint(blueprint, model_spec, output_prefix="unitree-go2-agentic")
        ```
    """

    default_config: type[EvalGeneratorConfig] = EvalGeneratorConfig
    config: EvalGeneratorConfig
    _llm: BaseChatModel | None = None
    _model_spec: ModelSpec | None = None

    def generate_from_blueprint(
        self, blueprint: ModuleBlueprintSet, model_spec: ModelSpec, output_prefix: str = "evals"
    ) -> Path | None:
        """Generate evaluation dataset from a DIMOS blueprint."""
        return self.generate_from_tools(
            extract_skills_from_blueprint(blueprint) or [], model_spec, output_prefix
        )

    def generate_from_tools(
        self, tools: ToolSchemaList, model_spec: ModelSpec, output_prefix: str = "evals"
    ) -> Path | None:
        """Generate evaluation dataset from a list of tool schemas."""
        if not tools:
            return None
        self._llm = init_chat_model(
            model_provider=model_spec.provider,
            model=model_spec.model,
            temperature=model_spec.temperature,
        )
        self._model_spec, all_evals = model_spec, []
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        both = self.config.include_single_turn and self.config.include_multi_turn
        if self.config.include_single_turn:
            all_evals.extend(
                self._generate_evals(
                    tools,
                    max(1, self.config.num_evals // 2) if both else self.config.num_evals,
                    True,
                )
            )
        if self.config.include_multi_turn:
            all_evals.extend(
                self._generate_evals(
                    tools,
                    max(1, self.config.num_evals - len(all_evals))
                    if both
                    else self.config.num_evals,
                    False,
                )
            )
        return self._write_json(all_evals, tools, output_prefix)

    def _generate_evals(
        self, tools: ToolSchemaList, num_evals: int, single_turn: bool
    ) -> list[dict[str, Any]]:
        """Generate evaluation examples using LLM."""
        evals, remaining = [], num_evals
        prompt = self.config.generation_prompt or (
            SINGLE_TURN_SYSTEM_PROMPT if single_turn else MULTI_TURN_SYSTEM_PROMPT
        )
        builder = build_single_turn_prompt if single_turn else build_multi_turn_prompt
        while remaining > 0:
            batch_size = min(self.config.batch_size, remaining)
            try:
                args = (tools, batch_size) + (
                    (self.config.max_turns_per_conversation,) if not single_turn else ()
                )
                content = getattr(
                    self._llm.invoke(
                        [SystemMessage(content=prompt), HumanMessage(content=builder(*args))]
                    ),
                    "content",
                    "",
                )
                evals.extend(
                    ev
                    for item in self._parse_json_response(content)
                    if (ev := self._format_eval(item, tools, single_turn))
                )
                remaining -= len(evals) - (num_evals - remaining)
            except Exception as e:
                logger.error(f"Error: {e}")
                remaining -= batch_size
        return evals[:num_evals]

    def _parse_json_response(self, content: str) -> list[dict[str, Any]]:
        """Parse JSON array from LLM response."""
        content = content.strip()
        for start in ["```json", "```"]:
            if start in content:
                content = content[
                    content.find(start) + len(start) : content.find(
                        "```", content.find(start) + len(start)
                    )
                ]
                break
        if "[" in content:
            content = content[content.find("[") : content.rfind("]") + 1]
        try:
            result = json.loads(content)
            return result if isinstance(result, list) else [result]
        except (json.JSONDecodeError, ValueError):
            return []

    def _format_eval(
        self, item: dict[str, Any], tools: ToolSchemaList, single_turn: bool
    ) -> dict[str, Any] | None:
        """Format evaluation example (single or multi-turn)."""
        if single_turn:
            user_query, tool_calls = item.get("user_query", ""), item.get("tool_calls", [])
            return (
                (
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": user_query},
                            {
                                "role": "assistant",
                                "tool_calls": [mk_tool_call(tc) for tc in tool_calls],
                            },
                        ],
                        "tools": tools,
                    }
                )
                if (user_query and tool_calls)
                else None
            )
        conversation = item.get("conversation", [])
        if not conversation:
            return None
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        for msg in conversation:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                assistant_msg = {"role": "assistant"} | ({"content": content} if content else {})
                if tool_calls := msg.get("tool_calls", []):
                    assistant_msg["tool_calls"] = [mk_tool_call(tc) for tc in tool_calls]
                messages.append(assistant_msg)
            elif role == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": msg.get("name", ""),
                        "content": content,
                    }
                )
        return {"messages": messages, "tools": tools}

    def _write_json(self, evals: list[dict[str, Any]], tools: ToolSchemaList, prefix: str) -> Path:
        """Write evals to JSON file."""
        filepath = (
            self.config.output_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filepath, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "num_evals": len(evals),
                        "model": self._model_spec.model if self._model_spec else None,
                        "provider": self._model_spec.provider if self._model_spec else None,
                        "tools": tools,
                    },
                    "evals": evals,
                },
                f,
                indent=2,
            )
        logger.info(f"Wrote {len(evals)} evals to {filepath}")
        return filepath


eval_generator = EvalGenerator.blueprint

__all__ = ["EvalGenerator", "eval_generator"]
