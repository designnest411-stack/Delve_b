"""
Google Gemini API Client (Free Tier Engine)
───────────────────────────────────────────
Async HTTP client for Google Gemini models with intelligent multi-model
fallback cascade, rate-limit resilience, and token usage accounting.

Key features:
  • Zero-cost operation targeting Google Gemini Free Tier
  • Multi-model fallback hierarchy (Flash-Lite 500 RPD -> Gemma 14.4k RPD -> Flash -> Pro)
  • Automatic 429 Rate Limit cooldown tracking & model rotation
  • RPM rate pacing to prevent burst rate limits
  • High-capacity token generation (up to 8192 tokens per call)
  • Native JSON response steering and system instructions
"""

import asyncio
import logging
import time
from typing import Optional, List, Any

import httpx

from app.core.config import settings
from app.core.llm_budget import record_usage, reserve_llm_call

logger = logging.getLogger("research_agent.llm")


class GeminiClient:
    """Async client for Google Gemini v1beta REST API with multi-model fallback."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        self._model_cooldowns: dict[str, float] = {}
        self._model_timestamps: dict[str, list[float]] = {}
        self._last_request_time: float = 0.0
        self._pacer_lock = asyncio.Lock()

    def _get_models_cascade(self) -> List[str]:
        """
        Return an intelligently prioritized cascade of models:
        1. Balances load between gemini-3.1-flash-lite and gemini-3.5-flash-lite (15 RPM / 500 RPD each).
        2. Automatically routes to whichever workhorse model has lower request load in the last 60s.
        3. Excludes models in active cooldown.
        4. Cascades to gemini-flash-lite-latest, then reasoning models as backup.
        """
        now = time.time()
        # Clean expired cooldowns
        self._model_cooldowns = {m: exp for m, exp in self._model_cooldowns.items() if exp > now}

        # Clean timestamps older than 60s
        for m in list(self._model_timestamps.keys()):
            self._model_timestamps[m] = [t for t in self._model_timestamps[m] if now - t < 60.0]

        # Primary high-quota workhorse models (500 RPD, 15 RPM each)
        workhorses = [
            settings.gemini_primary_model,
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
        ]
        seen = set()
        dedup_workhorses = []
        for m in workhorses:
            if m and m not in seen:
                seen.add(m)
                dedup_workhorses.append(m)

        # Other models (20 RPD, used strictly as emergency fallback)
        other_fallbacks = [m for m in settings.gemini_fallback_models if m not in seen]

        # Prioritize workhorses not in cooldown, sorted by fewest requests in the last 60s
        avail_workhorses = [m for m in dedup_workhorses if m not in self._model_cooldowns]
        avail_workhorses.sort(key=lambda m: len(self._model_timestamps.get(m, [])))

        # Other available fallbacks
        avail_others = [m for m in other_fallbacks if m not in self._model_cooldowns]

        available = avail_workhorses + avail_others
        if not available:
            # If all are in cooldown, sort by earliest expiry
            all_models = dedup_workhorses + other_fallbacks
            available = sorted(all_models, key=lambda m: self._model_cooldowns.get(m, 0.0))

        return available

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(settings.llm_timeout_seconds)
                    )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Image Generation Fallback ──────────────────────────────────────────

    async def generate_image(self, prompt: str) -> str:
        """Image generation placeholder."""
        return ""

    # ── Core Generation Methods ────────────────────────────────────────────

    async def generate_content(
        self,
        prompt: str,
        response_mime_type: str = "text/plain",
        temperature: float = 0.4,
        max_output_tokens: int = 8192,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Send a prompt to Gemini and return the text response."""
        return await self._generate(
            system_prompt=None,
            user_prompt=prompt,
            response_mime_type=response_mime_type,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    async def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        response_mime_type: str = "text/plain",
        temperature: float = 0.4,
        max_output_tokens: int = 8192,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Generate content with a system instruction."""
        return await self._generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_mime_type=response_mime_type,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    async def _generate(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        response_mime_type: str,
        temperature: float,
        max_output_tokens: int,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        generation_config: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if response_mime_type == "application/json":
            generation_config["responseMimeType"] = "application/json"

        payload: dict = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": generation_config,
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        return await self._send_with_fallback(payload)

    # ── Multi-Model Fallback Engine ────────────────────────────────────────

    async def _send_with_fallback(self, payload: dict) -> str:
        """
        Send the request with automatic fallback across free tier models on 429
        (Rate Limit / Quota) with rate pacing and model cooldown tracking.
        """
        client = await self._get_client()
        semaphore = await reserve_llm_call()
        api_key = settings.effective_gemini_api_key

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured in backend/.env. "
                "Please add your free API key from https://aistudio.google.com/"
            )

        models = self._get_models_cascade()
        last_error = None

        try:
            for model_index, model in enumerate(models):
                endpoint = f"{settings.gemini_endpoint(model)}?key={api_key}"
                headers = {"Content-Type": "application/json"}

                for attempt in range(2):
                    # Rate pacer: ensure smooth intervals between calls
                    async with self._pacer_lock:
                        now = time.time()
                        elapsed = now - self._last_request_time
                        if elapsed < 1.2:
                            await asyncio.sleep(1.2 - elapsed)
                        self._last_request_time = time.time()

                    try:
                        resp = await client.post(endpoint, json=payload, headers=headers)

                        if resp.status_code == 200:
                            data = resp.json()
                            self._model_timestamps.setdefault(model, []).append(time.time())
                            usage = data.get("usageMetadata", {})
                            await record_usage({
                                "input_tokens": usage.get("promptTokenCount", 0),
                                "output_tokens": usage.get("candidatesTokenCount", 0),
                                "cache_read_input_tokens": 0,
                            })
                            text = self._extract_text(data)
                            if text:
                                return text
                            logger.warning("Gemini model %s returned empty text candidate.", model)

                        elif resp.status_code == 429:
                            last_error = f"HTTP 429 (Rate Limit on {model})"
                            # Put model on 60-second cooldown so subsequent calls don't hit it immediately
                            self._model_cooldowns[model] = time.time() + 60.0
                            logger.warning(
                                "Gemini model %s rate limited (HTTP 429). Placed on 60s cooldown. Rotating to next model...",
                                model
                            )
                            # Break inner attempt loop immediately to rotate model
                            break

                        elif resp.status_code in (500, 502, 503, 504):
                            last_error = f"HTTP {resp.status_code} on {model}"
                            logger.warning(
                                "Gemini model %s temporary server error (%s). Attempt %d/2.",
                                model, resp.status_code, attempt + 1
                            )
                            await asyncio.sleep(2.0)
                            continue

                        elif resp.status_code == 404:
                            last_error = f"HTTP 404 on {model}: {resp.text[:200]}"
                            self._model_cooldowns[model] = time.time() + 86400.0 * 7
                            logger.warning("Gemini model %s returned 404 (unavailable). Cooldown set for 7 days. Rotating to next model...", model)
                            break

                        else:
                            error_text = resp.text[:400]
                            last_error = f"HTTP {resp.status_code} on {model}: {error_text}"
                            logger.error("Gemini API error (%s): %s", model, error_text)
                            # Don't retry client errors (400, 403) on same model, try fallback
                            break

                    except httpx.TimeoutException:
                        last_error = f"Timeout on {model}"
                        logger.warning("Gemini request to %s timed out. Switching to fallback model...", model)
                        break

                    except Exception as e:
                        last_error = f"{type(e).__name__} on {model}: {e}"
                        logger.error("Gemini request exception on %s: %s", model, e)
                        break

            raise RuntimeError(f"Gemini API failed across all fallback models: {last_error}")

        finally:
            if semaphore:
                semaphore.release()

    @staticmethod
    def _extract_text(response_data: dict) -> str:
        """Extract generated text from Gemini candidate parts."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return ""
            first = candidates[0]
            content = first.get("content", {})
            parts = content.get("parts", [])
            text_parts = [part.get("text", "") for part in parts if part.get("text")]
            return "".join(text_parts).strip()
        except (KeyError, IndexError, AttributeError) as exc:
            logger.debug("Error extracting text from Gemini response: %s", exc)
            return ""

    # ── API Key Validation ────────────────────────────────────────────────

    async def validate_api_key(self) -> bool:
        """Validate the Gemini API key with a fast, minimal ping call."""
        api_key = settings.effective_gemini_api_key
        if not api_key:
            logger.warning("No GEMINI_API_KEY found in configuration.")
            return False

        model = settings.gemini_primary_model
        endpoint = f"{settings.gemini_endpoint(model)}?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 4},
        }

        client = await self._get_client()
        try:
            resp = await client.post(endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=15.0)
            if resp.status_code in (200, 429):
                logger.info("Gemini API key validated successfully (model=%s, HTTP %d).", model, resp.status_code)
                return True
            logger.error("Gemini API key validation failed (HTTP %s): %s", resp.status_code, resp.text[:300])
            return False
        except Exception as e:
            logger.error("Gemini API key validation exception: %s", e)
            return False


# Singleton instance
llm_client = GeminiClient()
