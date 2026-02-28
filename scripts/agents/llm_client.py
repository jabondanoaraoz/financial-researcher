"""
LLM Client
==========
Provider-agnostic LLM wrapper used by all AI-driven agents.

Supports: Groq · Anthropic · OpenAI · DeepSeek · Ollama (local)

Provider selection and credentials are read from environment variables:
    LLM_PROVIDER  → groq | anthropic | openai | deepseek | ollama
    LLM_MODEL     → optional override (falls back to per-provider defaults)

Author: Financial Researcher Team
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default model per provider (update as providers deprecate models)
DEFAULT_MODELS = {
    "groq":      "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai":    "gpt-4o-mini",
    "deepseek":  "deepseek-chat",
    "ollama":    "llama3.1",
}


class LLMClient:
    """
    Thin, provider-agnostic wrapper around chat-completion APIs.

    Usage
    -----
    >>> client = LLMClient()                          # reads from env
    >>> text   = client.generate(system, user)        # plain text
    >>> obj    = client.generate(system, user,        # structured JSON
    ...              response_format={"type": "json_object"})

    Parameters
    ----------
    provider : str
        LLM backend.  Defaults to LLM_PROVIDER env var, then "groq".
    model : str | None
        Specific model override.  Defaults to LLM_MODEL env var, then
        the provider default from DEFAULT_MODELS.
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    max_tokens : int
        Maximum tokens in the generated response.
    """

    def __init__(
        self,
        provider: Optional[str]   = None,
        model: Optional[str]      = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int]  = None,
    ):
        self.provider    = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        self.model       = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(self.provider, "llama-3.3-70b-versatile")
        self.temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.max_tokens  = max_tokens  or int(os.getenv("LLM_MAX_TOKENS", "4000"))

        self._client = self._init_client()
        logger.info(f"LLMClient ready: provider={self.provider}, model={self.model}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_client(self):
        """Instantiate the underlying SDK client for the chosen provider."""
        if self.provider == "groq":
            return self._init_groq()
        elif self.provider == "anthropic":
            return self._init_anthropic()
        elif self.provider == "openai":
            return self._init_openai()
        elif self.provider == "deepseek":
            return self._init_deepseek()
        elif self.provider == "ollama":
            return self._init_ollama()
        else:
            raise ValueError(f"Unknown LLM provider: '{self.provider}'. "
                             f"Valid options: {list(DEFAULT_MODELS)}")

    def _init_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set")
        from groq import Groq
        return Groq(api_key=api_key)

    def _init_anthropic(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_claude_api_key_here":
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    def _init_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    def _init_deepseek(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key == "your_deepseek_key_here":
            raise EnvironmentError("DEEPSEEK_API_KEY not set")
        # DeepSeek is OpenAI-compatible
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def _init_ollama(self):
        # Uses openai-compatible endpoint at localhost
        from openai import OpenAI
        return OpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Generate a text response from the LLM.

        Parameters
        ----------
        system_prompt : str
            Sets the persona/context for the model.
        user_prompt : str
            The actual question or analysis request.
        response_format : dict | None
            Pass {"type": "json_object"} to request structured JSON output.
            Not all providers support this — Groq and OpenAI do.

        Returns
        -------
        str
            Raw model output (plain text or JSON string).
        """
        try:
            if self.provider == "anthropic":
                return self._generate_anthropic(system_prompt, user_prompt)
            else:
                return self._generate_openai_compatible(
                    system_prompt, user_prompt, response_format
                )
        except Exception as exc:
            logger.error(f"LLM generation failed ({self.provider}): {exc}")
            raise

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Convenience wrapper that requests JSON and parses the response.

        Returns an empty dict on parse failure (agent logic handles it).
        """
        raw = self.generate(
            system_prompt,
            user_prompt,
            response_format={"type": "json_object"}
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract JSON block if the model added prose around it
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning("LLM returned non-JSON response, returning empty dict")
            return {}

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    def _generate_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[dict],
    ) -> str:
        """Handles Groq, OpenAI, DeepSeek, Ollama (all use the same SDK)."""
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if response_format and self.provider in ("groq", "openai"):
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Anthropic SDK uses a different interface."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
