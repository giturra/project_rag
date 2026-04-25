# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import os
# import sys
# from dataclasses import asdict
# from typing import Optional, Sequence

# from llm.types import Message
# from llm.config import LLMConfig

# # If you already have llm/factory.py, keep this import:
# try:
#     from llm.factory import build_llm
# except Exception:
#     build_llm = None


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser("llm_test_cli")

#     # Backend selection
#     p.add_argument("--backend", choices=["openai", "ollama", "dummy"], default="openai")
#     p.add_argument("--model", required=True, help="Model name, e.g. gpt-3.5-turbo-16k")

#     # Connection / auth
#     p.add_argument("--base-url", default=None, help="Optional base URL (proxy/host).")
#     p.add_argument("--api-key", default=None, help="API key (prefer env OPENAI_API_KEY).")
#     p.add_argument("--timeout-s", type=float, default=60.0)

#     # Generation params
#     p.add_argument("--temperature", type=float, default=0.0)
#     p.add_argument("--max-tokens", type=int, default=128)
#     p.add_argument("--stop", nargs="*", default=None, help="Stop strings (space-separated).")

#     # Prompt
#     p.add_argument("--system", default="You are a concise assistant.")
#     p.add_argument("--prompt", required=True, help="User prompt text.")
#     p.add_argument("--print-raw", action="store_true", help="Print raw backend response object.")

#     return p.parse_args()


# def resolve_api_key(cli_key: Optional[str], backend: str) -> Optional[str]:
#     if cli_key:
#         return cli_key
#     if backend == "openai":
#         return os.getenv("OPENAI_API_KEY")
#     return None


# def main() -> int:
#     args = parse_args()

#     api_key = resolve_api_key(args.api_key, args.backend)

#     cfg = LLMConfig(
#         backend=args.backend,  # type: ignore
#         model=args.model,
#         api_key=api_key,
#         base_url=args.base_url,
#         timeout_s=args.timeout_s,
#     )

#     # Build the LLM (prefer your factory)
#     if build_llm is not None:
#         llm = build_llm(cfg)
#     else:
#         raise RuntimeError("LLM factory not available.")

#     messages: Sequence[Message] = [
#         Message("system", args.system),
#         Message("user", args.prompt),
#     ]

#     result = llm.generate(
#         messages,
#         temperature=args.temperature,
#         max_tokens=args.max_tokens,
#         stop=args.stop,
#     )

#     print("\n=== CONFIG (sanitized) ===")
#     cfg_dict = asdict(cfg)
#     if "api_key" in cfg_dict and cfg_dict["api_key"]:
#         cfg_dict["api_key"] = "***"
#     print(cfg_dict)

#     print("\n=== OUTPUT ===")
#     print(result.text)

#     print("\n=== METADATA ===")
#     print("backend:", getattr(result, "backend", ""))
#     print("model:", getattr(result, "model", ""))
#     print("latency_ms:", round(getattr(result, "latency_ms", 0.0), 2))
#     print("usage:", getattr(result, "usage", None))

#     if args.print_raw:
#         print("\n=== RAW ===")
#         print(result.raw)

#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())


from dotenv import load_dotenv, find_dotenv
import os

# ---- your existing code ----
from prompts.direct import build_messages
from llm.factory import build_llm
from llm.config import LLMConfig


# Load .env BEFORE anything else
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    raise FileNotFoundError(".env file not found. Run from project root.")
load_dotenv(dotenv_path)

# (Optional but recommended sanity check)
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"


cfg = LLMConfig(
    backend="openai",
    model="gpt-3.5-turbo-16k",
)

llm = build_llm(cfg)

messages = build_messages("In what year was the university where Sergei Aleksandrovich Tokarev was a professor founded?")

result = llm.generate(messages, temperature=0.2)
print(result.text)
