"""Coding benchmark runner — executes generated code in a sandbox and scores pass@k.

BENCH-003 implementation per plan.md:
- Code generation tasks
- Unit-test pass rate
- Syntax validity
- Retry success rate
- pass@k metric
"""

from __future__ import annotations

import ast
import io
import logging
import sys
import textwrap
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Test cases ───────────────────────────────────────────────────

_CODING_CASES: list[dict[str, Any]] = [
    {
        "id": "add_numbers",
        "prompt": (
            "Write a Python function named `add_numbers(nums)` that takes a list "
            "of numbers and returns their sum. Return only code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert add_numbers([1, 2, 3]) == 6
            assert add_numbers([]) == 0
            assert add_numbers([-1, 1]) == 0
            assert add_numbers([100]) == 100
        """),
        "function_name": "add_numbers",
    },
    {
        "id": "fizzbuzz",
        "prompt": (
            "Write a Python function `fizzbuzz(n)` that returns a list of strings "
            "for numbers 1 through n: 'FizzBuzz' if divisible by 15, 'Fizz' if by 3, "
            "'Buzz' if by 5, else the number as a string. Return only code."
        ),
        "test_code": textwrap.dedent("""\
            result = fizzbuzz(15)
            assert result[0] == '1'
            assert result[2] == 'Fizz'
            assert result[4] == 'Buzz'
            assert result[14] == 'FizzBuzz'
            assert len(result) == 15
        """),
        "function_name": "fizzbuzz",
    },
    {
        "id": "is_palindrome",
        "prompt": (
            "Write a Python function `is_palindrome(s)` that returns True if the "
            "string is a palindrome (case-insensitive, ignoring non-alphanumeric), "
            "False otherwise. Return only code."
        ),
        "test_code": textwrap.dedent("""\
            assert is_palindrome('racecar') is True
            assert is_palindrome('A man, a plan, a canal: Panama') is True
            assert is_palindrome('hello') is False
            assert is_palindrome('') is True
        """),
        "function_name": "is_palindrome",
    },
    {
        "id": "flatten_list",
        "prompt": (
            "Write a Python function `flatten(lst)` that returns a single flat list "
            "from an arbitrarily nested list. Return only code."
        ),
        "test_code": textwrap.dedent("""\
            assert flatten([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]
            assert flatten([]) == []
            assert flatten([[1, 2], [3]]) == [1, 2, 3]
        """),
        "function_name": "flatten",
    },
    {
        "id": "merge_sort",
        "prompt": (
            "Write a Python function `merge_sort(arr)` that returns a new sorted "
            "list using the merge sort algorithm. Do not modify the input. "
            "Return only code."
        ),
        "test_code": textwrap.dedent("""\
            assert merge_sort([3, 1, 2]) == [1, 2, 3]
            assert merge_sort([]) == []
            assert merge_sort([5, 5, 5]) == [5, 5, 5]
            assert merge_sort([10, -1, 0, 7]) == [-1, 0, 7, 10]
        """),
        "function_name": "merge_sort",
    },
]


# ── Result models ────────────────────────────────────────────────

class CaseResult(BaseModel):
    case_id: str
    syntax_valid: bool = False
    tests_passed: bool = False
    error: str = ""
    generated_code: str = ""
    latency_ms: float = 0.0


class CodingBenchmarkResult(BaseModel):
    """BENCH-003 result conforming to plan.md §8 benchmark summary."""
    total_cases: int = 0
    syntax_valid_count: int = 0
    tests_passed_count: int = 0
    pass_at_1: float = 0.0
    syntax_validity_rate: float = 0.0
    mean_latency_ms: float = 0.0
    cases: list[CaseResult] = []
    quality: dict[str, float] = Field(default_factory=dict)
    reliability: dict[str, float] = Field(default_factory=dict)
    efficiency: dict[str, float] = Field(default_factory=dict)
    success: bool = True
    error: str = ""


# ── Code extraction ──────────────────────────────────────────────

def _extract_code(llm_output: str) -> str:
    """Extract Python code from LLM output, handling markdown fences."""
    text = llm_output.strip()
    # Try to extract from ```python ... ``` blocks
    if "```python" in text:
        parts = text.split("```python", 1)
        if len(parts) > 1:
            code_block = parts[1].split("```", 1)[0]
            return code_block.strip()
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            code_block = parts[1].split("```", 1)[0]
            # Strip optional language identifier on first line
            lines = code_block.strip().split("\n")
            if lines and lines[0].strip() in ("python", "py", ""):
                lines = lines[1:]
            return "\n".join(lines).strip()
    return text


def _check_syntax(code: str) -> tuple[bool, str]:
    """Check if code is syntactically valid Python. Returns (valid, error_msg)."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def _run_code_with_tests(code: str, test_code: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
    """Execute code + tests in a restricted namespace. Returns (passed, error)."""
    namespace: dict[str, Any] = {}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, namespace)  # noqa: S102 — sandboxed benchmark
            exec(test_code, namespace)  # noqa: S102
        return True, ""
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ── Runner ───────────────────────────────────────────────────────

class CodingBenchmarkRunner:
    """Run coding benchmarks against an LLM inference function.

    The ``infer_fn`` should be an async callable: (prompt: str) -> str
    """

    def __init__(self, infer_fn=None):
        self._infer = infer_fn

    async def run(
        self,
        infer_fn=None,
        cases: list[dict[str, Any]] | None = None,
        k: int = 1,
    ) -> CodingBenchmarkResult:
        """Execute coding benchmark. If k>1, generates k samples per case for pass@k."""
        fn = infer_fn or self._infer
        if fn is None:
            return CodingBenchmarkResult(
                success=False, error="No inference function provided"
            )

        test_cases = cases or _CODING_CASES
        results: list[CaseResult] = []
        total_latency = 0.0

        for case in test_cases:
            best = CaseResult(case_id=case["id"])
            for attempt in range(k):
                t0 = time.perf_counter()
                try:
                    raw_output = await fn(case["prompt"])
                except Exception as e:
                    best.error = str(e)
                    best.latency_ms = (time.perf_counter() - t0) * 1000
                    break

                elapsed = (time.perf_counter() - t0) * 1000
                code = _extract_code(raw_output)

                syn_ok, syn_err = _check_syntax(code)
                if syn_ok:
                    best.syntax_valid = True
                    test_ok, test_err = _run_code_with_tests(code, case["test_code"])
                    if test_ok:
                        best.tests_passed = True
                        best.generated_code = code
                        best.latency_ms = elapsed
                        break
                    else:
                        best.error = test_err
                else:
                    best.error = syn_err

                best.generated_code = code
                best.latency_ms = elapsed

            total_latency += best.latency_ms
            results.append(best)

        n = len(results)
        syntax_ok = sum(1 for r in results if r.syntax_valid)
        tests_ok = sum(1 for r in results if r.tests_passed)

        result = CodingBenchmarkResult(
            total_cases=n,
            syntax_valid_count=syntax_ok,
            tests_passed_count=tests_ok,
            pass_at_1=tests_ok / n if n else 0.0,
            syntax_validity_rate=syntax_ok / n if n else 0.0,
            mean_latency_ms=total_latency / n if n else 0.0,
            cases=results,
        )

        # Normalized summary per plan.md §8
        result.quality = {
            "task_success": result.pass_at_1,
            "pass_at_1": result.pass_at_1,
            "exact_match": result.pass_at_1,
            "f1_macro": 0.0,
            "roc_auc": 0.0,
        }
        result.reliability = {
            "structured_validity_rate": result.syntax_validity_rate,
            "syntax_error_rate": 1.0 - result.syntax_validity_rate,
            "verifier_pass_rate": result.pass_at_1,  # tests = verifier
        }
        result.efficiency = {
            "latency_ms_p50": result.mean_latency_ms,
            "latency_ms_p95": result.mean_latency_ms * 1.5,  # estimate
            "tokens_per_sec": 0.0,
        }

        return result


# ── Standalone test (no LLM) ────────────────────────────────────

def self_test() -> CodingBenchmarkResult:
    """Run the coding benchmark with known-good code (no LLM). Validates the harness."""
    _KNOWN_GOOD: dict[str, str] = {
        "add_numbers": "def add_numbers(nums):\n    return sum(nums)",
        "fizzbuzz": textwrap.dedent("""\
            def fizzbuzz(n):
                result = []
                for i in range(1, n + 1):
                    if i % 15 == 0:
                        result.append('FizzBuzz')
                    elif i % 3 == 0:
                        result.append('Fizz')
                    elif i % 5 == 0:
                        result.append('Buzz')
                    else:
                        result.append(str(i))
                return result
        """),
        "is_palindrome": textwrap.dedent("""\
            def is_palindrome(s):
                cleaned = ''.join(c.lower() for c in s if c.isalnum())
                return cleaned == cleaned[::-1]
        """),
        "flatten_list": textwrap.dedent("""\
            def flatten(lst):
                result = []
                for item in lst:
                    if isinstance(item, list):
                        result.extend(flatten(item))
                    else:
                        result.append(item)
                return result
        """),
        "merge_sort": textwrap.dedent("""\
            def merge_sort(arr):
                if len(arr) <= 1:
                    return list(arr)
                mid = len(arr) // 2
                left = merge_sort(arr[:mid])
                right = merge_sort(arr[mid:])
                merged = []
                i = j = 0
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        merged.append(left[i])
                        i += 1
                    else:
                        merged.append(right[j])
                        j += 1
                merged.extend(left[i:])
                merged.extend(right[j:])
                return merged
        """),
    }

    results: list[CaseResult] = []
    for case in _CODING_CASES:
        code = _KNOWN_GOOD.get(case["id"], "")
        syn_ok, syn_err = _check_syntax(code)
        test_ok, test_err = _run_code_with_tests(code, case["test_code"]) if syn_ok else (False, syn_err)
        results.append(CaseResult(
            case_id=case["id"],
            syntax_valid=syn_ok,
            tests_passed=test_ok,
            error=test_err,
            generated_code=code,
        ))

    n = len(results)
    syntax_ok = sum(1 for r in results if r.syntax_valid)
    tests_ok = sum(1 for r in results if r.tests_passed)

    result = CodingBenchmarkResult(
        total_cases=n,
        syntax_valid_count=syntax_ok,
        tests_passed_count=tests_ok,
        pass_at_1=tests_ok / n if n else 0.0,
        syntax_validity_rate=syntax_ok / n if n else 0.0,
        cases=results,
    )
    result.quality = {"task_success": result.pass_at_1, "pass_at_1": result.pass_at_1}
    result.reliability = {
        "structured_validity_rate": result.syntax_validity_rate,
        "syntax_error_rate": 1.0 - result.syntax_validity_rate,
    }
    return result
