"""run.sh, on the bash macOS actually ships.

That is bash **3.2.57**, frozen there since 2007 for licensing reasons, and it
differs from any modern bash in ways that only appear at runtime. run.sh sets
`set -euo pipefail`, and under `set -u` bash 3.2 treats the expansion of an
*empty* array as an unbound variable — bash 4.4 fixed it, and macOS will not be
shipping bash 4.4.

So `./run.sh` with no arguments, which is the ordinary way to start the stack,
died with "ARGS[@]: unbound variable" the moment argument parsing was added.
`bash -n` does not catch it; nothing catches it but running the thing.

These drive the parsing block out of the real file rather than a copy, so the
copy cannot drift away from what ships.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

RUN_SH = Path(__file__).resolve().parents[1] / "run.sh"


def parse(*arguments: str) -> tuple[str, int, str]:
    """Run run.sh's argument parsing under bash, and report what it decided."""
    text = RUN_SH.read_text()
    block = re.search(r"^ARGS=\(\).*?^set -- .*?$", text, re.MULTILINE | re.DOTALL)
    assert block, "the argument-parsing block moved; this test needs updating"

    script = (
        "set -euo pipefail\n"
        + block.group(0)
        + '\necho "$RELOAD|$#|$*"\n'
    )
    done = subprocess.run(
        ["bash", "-c", script, "run.sh", *arguments],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    reload_flag, count, rest = done.stdout.strip().split("|", 2)
    return reload_flag, int(count), rest


def test_the_script_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", str(RUN_SH)]).returncode == 0


def test_no_arguments_at_all_parses():
    """The case that broke, and the most common way anyone runs this."""
    flag, count, _ = parse()
    assert flag == "0" and count == 0


def test_reload_alone_parses():
    flag, count, _ = parse("--reload")
    assert flag == "1" and count == 0


def test_a_target_survives():
    flag, count, rest = parse("dev")
    assert (flag, count, rest) == ("0", 1, "dev")


def test_reload_is_stripped_from_the_target():
    """Otherwise the case statement sees "--reload" as the target and prints
    usage instead of starting a worker."""
    flag, count, rest = parse("worker-agents", "--reload")
    assert flag == "1"
    assert rest == "worker-agents"


def test_reload_may_come_first():
    flag, _, rest = parse("--reload", "worker-agents")
    assert flag == "1" and rest == "worker-agents"


def test_extra_arguments_reach_their_target():
    """`./run.sh mcp --http` — the mcp target forwards what follows it."""
    _, count, rest = parse("mcp", "--http")
    assert count == 2 and rest == "mcp --http"


def test_reload_does_not_eat_a_targets_own_flags():
    flag, count, rest = parse("--reload", "mcp", "--http")
    assert flag == "1"
    assert rest == "mcp --http"


def test_an_argument_containing_a_space_is_not_split():
    _, count, _ = parse("some target")
    assert count == 1


def test_the_empty_array_expansion_is_guarded():
    """Belt and braces on the specific construct.

    A plain "${ARGS[@]}" is what broke, and it breaks only on bash 3.2 with an
    empty array — so a contributor on Linux would never see it, and the tests
    above would still pass if they ran under bash 5.
    """
    text = RUN_SH.read_text()
    assert 'set -- "${ARGS[@]}"' not in text
    assert 'set -- ${ARGS[@]+"${ARGS[@]}"}' in text


@pytest.mark.parametrize(
    "target",
    ["all", "dev", "worker-io", "worker-cpu", "worker-agents", "worker-build",
     "api", "dashboard", "flower", "mcp"],
)
def test_every_documented_target_is_handled(target):
    """The usage line and the case statement drifting apart is a target that
    prints usage instead of running."""
    text = RUN_SH.read_text()
    case_block = text[text.index('case "${1:-all}" in'):]
    assert f"{target})" in case_block or f"|{target})" in case_block
    assert target in text[text.index("usage:"):text.index("usage:") + 300]
