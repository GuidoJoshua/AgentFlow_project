import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


@dataclass
class ToolCandidate:
    tool_name: str
    score: float
    query_score: float
    step_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "tool_name": self.tool_name,
            "score": self.score,
            "query_score": self.query_score,
            "step_score": self.step_score,
        }


@dataclass
class ToolSelectionResult:
    selected_tool: str
    selected_score: float
    candidates: List[ToolCandidate]
    embedding_model: str
    query_text: str
    step_text: str
    memory_text: str
    query_weight: float
    step_weight: float
    query_task_type: str
    document_task_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": "embedding_similarity",
            "selected_tool": self.selected_tool,
            "selected_score": self.selected_score,
            "embedding_model": self.embedding_model,
            "query_weight": self.query_weight,
            "step_weight": self.step_weight,
            "query_task_type": self.query_task_type,
            "document_task_type": self.document_task_type,
            "query_text": self.query_text,
            "step_text": self.step_text,
            "memory_text": self.memory_text,
            "top_candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class ToolSelector:
    def __init__(
        self,
        toolbox_metadata: Optional[dict] = None,
        available_tools: Optional[List[str]] = None,
        embedding_model_name: str = "gemini-embedding-001",
        verbose: bool = False,
        api_key: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
        top_k: int = 3,
        query_weight: float = 0.35,
        step_weight: float = 0.65,
        query_task_type: str = "RETRIEVAL_QUERY",
        document_task_type: str = "RETRIEVAL_DOCUMENT",
        max_retries: int = 3,
        max_query_characters: int = 4000,
        max_step_characters: int = 5000,
        max_memory_characters: int = 3500,
        max_memory_actions: int = 4,
        max_profile_characters: int = 3000,
    ):
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.embedding_model_name = embedding_model_name
        self.verbose = verbose
        self.output_dimensionality = output_dimensionality
        self.top_k = top_k
        self.query_weight = query_weight
        self.step_weight = step_weight
        self.query_task_type = query_task_type
        self.document_task_type = document_task_type
        self.max_retries = max_retries
        self.max_query_characters = max_query_characters
        self.max_step_characters = max_step_characters
        self.max_memory_characters = max_memory_characters
        self.max_memory_actions = max_memory_actions
        self.max_profile_characters = max_profile_characters

        self.client = None
        self.tool_profiles: Dict[str, str] = {}
        self.tool_embeddings: Dict[str, List[float]] = {}

        self._refresh_tool_profiles()

        if len(self.available_tools) <= 1:
            return

        if genai is None or types is None:
            raise ImportError(
                "google-genai is required for embedding-based tool selection. "
                "Please install the package or switch tool_selection_mode to 'planner'."
            )

        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Embedding-based tool selection requires GOOGLE_API_KEY or GEMINI_API_KEY."
            )

        self.client = genai.Client(api_key=resolved_api_key)
        self._refresh_tool_embeddings()

    def _refresh_tool_profiles(self) -> None:
        self.tool_profiles = {
            tool_name: self._build_tool_profile(tool_name, self.toolbox_metadata.get(tool_name, {}))
            for tool_name in self.available_tools
        }

    def _refresh_tool_embeddings(self) -> None:
        if not self.available_tools:
            self.tool_embeddings = {}
            return

        profile_texts = [self.tool_profiles[tool_name] for tool_name in self.available_tools]
        embeddings = self._embed_texts(profile_texts, task_type=self.document_task_type)
        self.tool_embeddings = {
            tool_name: embedding
            for tool_name, embedding in zip(self.available_tools, embeddings)
        }

    def _build_tool_profile(self, tool_name: str, metadata: Dict[str, Any]) -> str:
        description = metadata.get("tool_description", "")
        output_type = metadata.get("output_type", "")
        input_types = metadata.get("input_types", {}) or {}
        user_metadata = metadata.get("user_metadata", {}) or {}
        best_practices = user_metadata.get("best_practices", "")
        limitations = user_metadata.get("limitations", "")
        demo_commands = metadata.get("demo_commands", []) or []

        lines = [
            f"Tool Name: {tool_name}",
            f"Description: {description}",
        ]

        if input_types:
            lines.append("Inputs:")
            for key, value in input_types.items():
                lines.append(f"- {key}: {value}")

        if output_type:
            lines.append(f"Output: {output_type}")

        if best_practices:
            lines.append("Best Practices:")
            lines.append(best_practices.strip())

        if limitations:
            lines.append("Limitations:")
            lines.append(limitations.strip())

        if demo_commands:
            example_descriptions = []
            for command in demo_commands[:3]:
                description_text = command.get("description") or command.get("command")
                if description_text:
                    example_descriptions.append(f"- {description_text}")
            if example_descriptions:
                lines.append("Representative Tasks:")
                lines.extend(example_descriptions)

        profile = "\n".join(lines)
        return self._truncate_text(profile, self.max_profile_characters)

    def _build_query_text(self, question: str, query_analysis: str) -> str:
        parts = [f"User Query:\n{(question or '').strip()}"]
        if query_analysis:
            parts.append(f"High-Level Analysis:\n{query_analysis.strip()}")
        return self._truncate_text("\n\n".join(parts), self.max_query_characters)

    def _stringify_memory_value(self, value: Any, max_characters: int = 400) -> str:
        text = str(value).strip()
        return self._truncate_text(text, max_characters)

    def _build_memory_text(self, memory_actions: Optional[Dict[str, Dict[str, Any]]]) -> str:
        if not memory_actions:
            return ""

        action_items = list(memory_actions.items())[-self.max_memory_actions :]
        lines = ["Recent Memory:"]

        for step_name, action in action_items:
            if not isinstance(action, dict):
                lines.append(f"- {step_name}: {self._stringify_memory_value(action)}")
                continue

            tool_name = action.get("tool_name", "Unknown_Tool")
            sub_goal = self._stringify_memory_value(action.get("sub_goal", ""), 200)
            result = self._stringify_memory_value(action.get("result", ""), 500)
            lines.append(f"- {step_name}")
            lines.append(f"  Tool: {tool_name}")
            if sub_goal:
                lines.append(f"  Sub-Goal: {sub_goal}")
            if result:
                lines.append(f"  Result: {result}")

        return self._truncate_text("\n".join(lines), self.max_memory_characters)

    def _build_step_text(
        self,
        context: str,
        sub_goal: str,
        memory_actions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        parts = [f"Current Sub-Goal:\n{(sub_goal or '').strip()}"]
        if context:
            parts.append(f"Execution Context:\n{context.strip()}")
        memory_text = self._build_memory_text(memory_actions)
        if memory_text:
            parts.append(memory_text)
        return self._truncate_text("\n\n".join(parts), self.max_step_characters)

    def _truncate_text(self, text: str, max_characters: int) -> str:
        clean_text = (text or "").strip()
        if len(clean_text) <= max_characters:
            return clean_text
        return clean_text[: max_characters - 3] + "..."

    def _make_embed_config(self, task_type: str):
        config_kwargs = {}
        if task_type:
            config_kwargs["task_type"] = task_type
        if self.output_dimensionality:
            config_kwargs["output_dimensionality"] = self.output_dimensionality
        return types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

    def _extract_embedding_values(self, response: Any) -> List[List[float]]:
        if hasattr(response, "embeddings") and response.embeddings is not None:
            raw_embeddings = response.embeddings
        elif hasattr(response, "embedding") and response.embedding is not None:
            raw_embeddings = [response.embedding]
        else:
            raise ValueError("Embedding response did not contain 'embedding' or 'embeddings'.")

        embeddings = []
        for embedding in raw_embeddings:
            values = getattr(embedding, "values", None)
            if values is None and isinstance(embedding, dict):
                values = embedding.get("values")
            if values is None:
                raise ValueError("Embedding values were not found in the response.")
            embeddings.append(self._normalize_vector(list(values)))
        return embeddings

    def _embed_texts(self, texts: List[str], task_type: str) -> List[List[float]]:
        if not texts:
            return []

        if self.client is None:
            raise RuntimeError("Embedding client is not initialized.")

        config = self._make_embed_config(task_type)
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model_name,
                    contents=texts,
                    config=config,
                )
                embeddings = self._extract_embedding_values(response)
                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"Expected {len(texts)} embeddings, received {len(embeddings)}."
                    )
                return embeddings
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))

        if len(texts) > 1:
            return [self._embed_texts([text], task_type)[0] for text in texts]

        raise RuntimeError(
            f"Failed to embed text with {self.embedding_model_name}: {last_error}"
        )

    def _normalize_vector(self, values: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return float("-inf")
        return sum(x * y for x, y in zip(a, b))

    def select_tool(
        self,
        question: str,
        query_analysis: str,
        context: str,
        sub_goal: str,
        memory_actions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ToolSelectionResult:
        if not self.available_tools:
            raise ValueError("No available tools were provided to the ToolSelector.")

        query_text = self._build_query_text(question, query_analysis)
        memory_text = self._build_memory_text(memory_actions)
        step_text = self._build_step_text(context, sub_goal, memory_actions)

        if len(self.available_tools) == 1:
            only_tool = self.available_tools[0]
            only_candidate = ToolCandidate(
                tool_name=only_tool,
                score=1.0,
                query_score=1.0,
                step_score=1.0,
            )
            return ToolSelectionResult(
                selected_tool=only_tool,
                selected_score=1.0,
                candidates=[only_candidate],
                embedding_model=self.embedding_model_name,
                query_text=query_text,
                step_text=step_text,
                memory_text=memory_text,
                query_weight=self.query_weight,
                step_weight=self.step_weight,
                query_task_type=self.query_task_type,
                document_task_type=self.document_task_type,
            )

        query_embedding, step_embedding = self._embed_texts(
            [query_text, step_text],
            task_type=self.query_task_type,
        )

        candidates = []
        for tool_name in self.available_tools:
            tool_embedding = self.tool_embeddings[tool_name]
            query_score = self._cosine_similarity(query_embedding, tool_embedding)
            step_score = self._cosine_similarity(step_embedding, tool_embedding)
            total_score = (self.query_weight * query_score) + (self.step_weight * step_score)
            candidates.append(
                ToolCandidate(
                    tool_name=tool_name,
                    score=total_score,
                    query_score=query_score,
                    step_score=step_score,
                )
            )

        ranked_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        selected_candidate = ranked_candidates[0]
        top_candidates = ranked_candidates[: self.top_k]

        if self.verbose:
            summary = ", ".join(
                f"{candidate.tool_name}={candidate.score:.4f}" for candidate in top_candidates
            )
            print(f"ToolSelector ranking: {summary}")

        return ToolSelectionResult(
            selected_tool=selected_candidate.tool_name,
            selected_score=selected_candidate.score,
            candidates=top_candidates,
            embedding_model=self.embedding_model_name,
            query_text=query_text,
            step_text=step_text,
            memory_text=memory_text,
            query_weight=self.query_weight,
            step_weight=self.step_weight,
            query_task_type=self.query_task_type,
            document_task_type=self.document_task_type,
        )
