"""The ONLY file that talks to the LLM. Provider is chosen with LLM_PROVIDER in .env:

  LLM_PROVIDER=google      -> Gemini  (needs GOOGLE_API_KEY)
  LLM_PROVIDER=groq        -> Groq    (needs GROQ_API_KEY, e.g. LLM_MODEL=openai/gpt-oss-120b)
  LLM_PROVIDER=anthropic   -> Claude  (needs ANTHROPIC_API_KEY)
"""
import os
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from .config import settings

PLACEHOLDERS = {"", "your-key-here", "paste-your-gemini-key-here", "paste-your-groq-key-here"}
KEY_ENV = {"google": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class LLMError(RuntimeError):
    """Readable error for the UI/API instead of a stack trace."""


def _provider() -> str:
    p = settings.llm_provider.lower()
    if p not in KEY_ENV:
        raise LLMError(f"Unknown LLM_PROVIDER '{p}'. Use 'google', 'groq' or 'anthropic'.")
    return p


def has_key() -> bool:
    try:
        return os.getenv(KEY_ENV[_provider()], "") not in PLACEHOLDERS
    except LLMError:
        return False


def _check_key():
    env = KEY_ENV[_provider()]
    if os.getenv(env, "") in PLACEHOLDERS:
        raise LLMError(f"{env} is not set. Put your key in the .env file.")


@lru_cache(maxsize=8)
def get_chain(max_tokens: int):
    """messages -> chat model -> str, with retries on transient errors."""
    if _provider() == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=settings.llm_model, max_output_tokens=max_tokens,
                                     google_api_key=os.getenv("GOOGLE_API_KEY"), max_retries=4)
        return llm | StrOutputParser()

    if _provider() == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=settings.llm_model, max_tokens=max_tokens,
                       api_key=os.getenv("GROQ_API_KEY"), max_retries=4)
        return llm | StrOutputParser()

    import anthropic
    from langchain_anthropic import ChatAnthropic

    transient = (anthropic.RateLimitError, anthropic.APIConnectionError,
                 anthropic.APITimeoutError, anthropic.InternalServerError)
    llm = ChatAnthropic(model=settings.llm_model, max_tokens=max_tokens)
    return (llm | StrOutputParser()).with_retry(
        retry_if_exception_type=transient, stop_after_attempt=4, wait_exponential_jitter=True)


def _messages(system: str, user: str):
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _wrap(e: Exception) -> LLMError:
    name, msg = type(e).__name__, str(e)
    low = msg.lower()
    env = KEY_ENV.get(settings.llm_provider.lower(), "your key")
    if name.endswith("PermissionDeniedError") or name in {"PermissionDenied"} or "denied access" in low:
        return LLMError("The provider denied access to this key/project (403 PERMISSION_DENIED). "
                        "Create a new key from another account/project, or switch LLM_PROVIDER in .env.")
    if "api key" in low or "api_key" in low or name.endswith("AuthenticationError") or name == "Unauthenticated":
        return LLMError(f"The API key was rejected. Check {env} in .env.")
    if name.endswith("NotFoundError") or name == "NotFound" or ("not found" in low and "model" in low):
        return LLMError(f"Model '{settings.llm_model}' not found. Check LLM_MODEL in .env.")
    if name.endswith("RateLimitError") or name in {"ResourceExhausted", "TooManyRequests"} or "429" in msg or "quota" in low:
        return LLMError("Rate limit / free-tier quota reached. Wait a minute and retry (or lower usage).")
    if name in {"APIConnectionError", "APITimeoutError", "InternalServerError", "ServiceUnavailable"}:
        return LLMError("The LLM service is busy or unreachable. Please retry in a moment.")
    return LLMError(f"LLM call failed ({name}): {msg[:300]}")


def complete(system: str, user: str, max_tokens: int = 1200) -> str:
    _check_key()
    try:
        return get_chain(max_tokens).invoke(_messages(system, user))
    except LLMError:
        raise
    except Exception as e:  # noqa: BLE001
        raise _wrap(e) from e


def complete_batch(system: str, users: list[str], max_tokens: int = 1200) -> list[str]:
    """Run several independent prompts (LangChain .batch). Concurrency is low on free tiers."""
    _check_key()
    try:
        return get_chain(max_tokens).batch([_messages(system, u) for u in users],
                                           config={"max_concurrency": settings.llm_concurrency})
    except LLMError:
        raise
    except Exception as e:  # noqa: BLE001
        raise _wrap(e) from e


def stream(system: str, user: str, max_tokens: int = 1200):
    _check_key()
    try:
        yield from get_chain(max_tokens).stream(_messages(system, user))
    except Exception as e:  # noqa: BLE001
        raise _wrap(e) from e