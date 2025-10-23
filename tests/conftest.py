from __future__ import annotations

import sys
import threading
import trace
from collections import defaultdict
from pathlib import Path

import pytest


def _statement_lines(path: Path) -> set[int]:
    lines: set[int] = set()
    try:
        content = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return lines

    for index, raw in enumerate(content, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.add(index)
    return lines


class TraceCoveragePlugin:
    def __init__(self, config: pytest.Config, cov_option: str, report_option: str | None) -> None:
        self.config = config
        self.report_option = report_option
        self.tracer = trace.Trace(count=True, trace=False)
        base_path = Path(cov_option)
        if not base_path.is_absolute():
            base_path = (config.rootpath / base_path).resolve()
        self.base_path = base_path
        self.root_path = Path(config.rootpath).resolve()
        self._results: trace.CoverageResults | None = None

    def start(self) -> None:
        sys.settrace(self.tracer.globaltrace)
        threading.settrace(self.tracer.globaltrace)

    def stop(self) -> None:
        sys.settrace(None)
        threading.settrace(None)
        self._results = self.tracer.results()
        self._report()

    def _iter_relevant_files(self) -> list[Path]:
        if not self.base_path.exists():
            return []

        files: list[Path] = []
        for file_path in self.base_path.rglob("*.py"):
            resolved = file_path.resolve()
            try:
                relative = resolved.relative_to(self.root_path)
            except ValueError:
                continue

            if relative.parts and relative.parts[0] == "tests":
                continue

            files.append(resolved)

        return files

    def _report(self) -> None:
        assert self._results is not None
        counts: dict[str, set[int]] = defaultdict(set)
        for filename, lineno in self._results.counts:
            counts[filename].add(lineno)

        summaries: list[tuple[str, int, int, float, list[int]]] = []
        total_statements = 0
        total_executed = 0

        for file_path in self._iter_relevant_files():
            resolved = file_path.resolve()
            executed_lines = counts.get(str(resolved), set())
            if not executed_lines and str(resolved) not in counts:
                continue

            statements = _statement_lines(resolved)
            if not statements:
                continue

            executed = len(statements & executed_lines)
            missed_lines = sorted(statements - executed_lines)
            coverage_percent = 100.0 * executed / len(statements) if statements else 100.0

            try:
                relative_name = str(resolved.relative_to(self.root_path))
            except ValueError:
                relative_name = str(resolved)

            summaries.append((relative_name, len(statements), len(missed_lines), coverage_percent, missed_lines))
            total_statements += len(statements)
            total_executed += executed

        terminal = self.config.pluginmanager.get_plugin("terminalreporter")
        if terminal is None:
            return

        if not summaries:
            terminal.write_line("No files collected for coverage measurement.")
            return

        header = "Name"
        terminal.write_line("\n---------- coverage: platform python ----------")
        terminal.write_line(f"{header:<60} Stmts   Miss  Cover")
        for name, stmts, miss, cover, missing in sorted(summaries):
            line = f"{name:<60} {stmts:5d} {miss:6d} {cover:6.2f}%"
            if self.report_option == "term-missing" and missing:
                missing_text = ",".join(str(num) for num in missing)
                line = f"{line}   Missing: {missing_text}"
            terminal.write_line(line)

        overall = 100.0 * total_executed / total_statements if total_statements else 100.0
        terminal.write_line(
            f"TOTAL{'':<55} {total_statements:5d} {total_statements - total_executed:6d} {overall:6.2f}%"
        )


@pytest.hookimpl(tryfirst=True)
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--cov", action="store", default=None, help="Measure coverage for the given path")
    parser.addoption("--cov-report", action="store", default=None, help="Select coverage report format")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    cov_option = config.getoption("--cov")
    if not cov_option:
        return

    report_option = config.getoption("--cov-report")
    plugin = TraceCoveragePlugin(config, cov_option, report_option)
    config.trace_coverage_plugin = plugin  # type: ignore[attr-defined]
    plugin.start()


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    plugin: TraceCoveragePlugin | None = getattr(config, "trace_coverage_plugin", None)
    if plugin is not None:
        plugin.stop()
