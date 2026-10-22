"""The README stays current with the code: every command-line option is documented, no named option is stale, and
every repo link and cited line still resolves.

Options come from the parsers themselves. Every module `python -m` can run (a package's __main__.py, or a module
with a __main__ guard) is run in this process with ArgumentParser.parse_known_args stopped at its first call, which
hands back the parser it was about to parse with. A dispatcher that picks a module by its first argument (board,
prepare) is run once per key of its command table, as is a command with a COMMANDS table of its own (review seam), a
subcommand is followed into its subparser or into the parser it delegates to (gate score), and a command's options
are documented in the README's command table, in the row whose
command is the longest prefix of the parser's own. A row whose options cell links to a heading also takes the options
in the first column of the first table under it (review's Settings). External URLs are not checked, since CI runs
offline.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / 'README.md'
PROG = 'python -m '

# A module with a __main__ guard that parses no command line of its own, with why it is not a command
NOT_COMMANDS = {
    'prepare.isolation': 'the child process one native read runs in; its arguments come from run_isolated',
}

FLAG = re.compile(r'(?<![\w-])(--[a-z0-9][a-z0-9-]*)')


class _Parsed(Exception):
    """Raised in place of parsing, carrying the parser that was about to parse."""


def _stop(self, args=None, namespace=None):
    raise _Parsed(self)


def runnable_modules(root: Path = REPO) -> list[str]:
    """Every module `python -m` runs in the tree: packages with __main__.py and modules with a __main__ guard."""
    out = []
    packages = [p for p in sorted(root.iterdir()) if (p / '__init__.py').is_file() and p.name != 'tests']
    for path in sorted(f for package in packages for f in package.rglob('*.py')):
        rel = path.relative_to(root)
        if not all((root / Path(*rel.parts[:i]) / '__init__.py').exists() for i in range(1, len(rel.parts))):
            continue
        text = path.read_text(errors='replace')
        if path.name == '__main__.py':
            out.append('.'.join(rel.parts[:-1]))
        elif re.search(r'^if __name__ == [\'"]__main__[\'"]', text, re.M):
            out.append('.'.join(rel.with_suffix('').parts))
    return out


def _run(module: str, argv: list[str]):
    """(parser the module was about to parse with, or None, module globals) for `python -m module *argv`."""
    spec = importlib.util.find_spec(module)
    if spec.submodule_search_locations is not None:
        spec = importlib.util.find_spec(f'{module}.__main__')
    code = spec.loader.get_code(spec.name)
    ns = {'__name__': '__main__', '__file__': spec.origin, '__spec__': spec, '__builtins__': __builtins__,
          '__package__': spec.parent}
    out = io.StringIO()
    with mock.patch.object(argparse.ArgumentParser, 'parse_known_args', _stop), \
            mock.patch.object(sys, 'argv', [PROG + module, *argv]), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            exec(code, ns)
        except _Parsed as parsed:
            return parsed.args[0], ns
        except SystemExit:
            pass
    return None, ns


def _signature(parser: argparse.ArgumentParser) -> tuple:
    return parser.description, tuple((tuple(a.option_strings), a.dest) for a in parser._actions)


def _walk(module: str, argv: list[str], parser: argparse.ArgumentParser, found: dict, origin: dict) -> None:
    # a module its own package also runs (label.run is python -m label) is collected once, under the package's
    # command, which is run first
    first = origin.setdefault(_signature(parser), module)
    if first == module or not module.startswith(first + '.'):
        found.setdefault(parser.prog, parser)
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, sub in action.choices.items():
            delegated, _ = _run(module, [*argv, name])
            # a command that hands its subcommand to another module's parser (gate score) is that parser
            target = delegated if delegated is not None and delegated.prog != parser.prog else sub
            _walk(module, [*argv, name], target, found, origin)


def collect_parsers(root: Path = REPO) -> dict[str, argparse.ArgumentParser]:
    """{prog: parser} for every command and subcommand of every runnable module."""
    found: dict[str, argparse.ArgumentParser] = {}
    origin: dict[tuple, str] = {}
    for module in sorted(runnable_modules(root), key=lambda m: m.count('.')):
        if module in NOT_COMMANDS:
            continue
        parser, ns = _run(module, [])
        if parser is not None:
            _walk(module, [], parser, found, origin)
            # a command whose first argument can also pick a command of its own from COMMANDS (review seam)
            for name in ns.get('COMMANDS') or {}:
                sub, _ = _run(module, [name])
                assert sub is not None and sub.prog != parser.prog, f'{module} {name} parses no command line of its own'
                _walk(module, [name], sub, found, origin)
            continue
        # a dispatcher: its first argument picks a command from a table of names, such as COMMANDS or ADAPTERS
        tables = [v for k, v in ns.items() if k.isupper() and isinstance(v, dict) and v
                  and all(isinstance(key, str) for key in v)]
        commands = [key for table in tables for key in table]
        hits = 0
        for name in commands:
            parser, _ = _run(module, [name])
            if parser is not None:
                hits += 1
                _walk(module, [name], parser, found, origin)
        assert hits, (f'{module} runs as a command but parses no command line here; if it is not a command, add it '
                      f'to NOT_COMMANDS with the reason')
    return found


def options(parser: argparse.ArgumentParser) -> set[str]:
    """The options a parser takes, each by its long form when it has one, without --help."""
    out = set()
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or not action.option_strings:
            continue
        longs = [s for s in action.option_strings if s.startswith('--')]
        out.add(longs[0] if longs else action.option_strings[0])
    return out


def command(prog: str) -> str:
    """`python -m board serve` -> `board serve`."""
    return prog[len(PROG):] if prog.startswith(PROG) else prog


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r'<[^>]+>', '', heading).strip().lower()
    text = re.sub(r'[`*_\\]', '', text)
    return re.sub(r'[^a-z0-9 -]', '', text).replace(' ', '-')


def headings(text: str) -> list[tuple[int, str]]:
    """(line index, heading text) for every Markdown and HTML heading outside code blocks."""
    out, fenced = [], False
    for i, line in enumerate(text.splitlines()):
        if line.startswith('```'):
            fenced = not fenced
        elif not fenced:
            m = re.match(r'#{1,6} (.+)$', line) or re.match(r'\s*<h[1-6][^>]*>(.+?)</h[1-6]>', line)
            if m:
                out.append((i, m.group(1)))
    return out


def tables(text: str) -> list[tuple[int, list[list[str]]]]:
    """(line index of the header, body rows as lists of cells) for every Markdown table."""
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        if lines[i].startswith('|') and i + 1 < len(lines) and re.match(r'\|[-:| ]+\|$', lines[i + 1]):
            start, rows, i = i, [], i + 2
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            out.append((start, rows))
        else:
            i += 1
    return out


def command_table(text: str) -> dict[str, set[str]]:
    """{command: options} from the README's command table, the one whose header is | Command | Options |. An
    options cell that links to a heading names the first table under that heading, whose first column holds the
    options."""
    lines = text.splitlines()
    heads = {slug(h): i for i, h in headings(text)}
    all_tables = tables(text)
    found = {}
    for start, rows in all_tables:
        if [c.strip() for c in lines[start].strip().strip('|').split('|')][:2] != ['Command', 'Options']:
            continue
        for cells in rows:
            name = re.search(r'`([^`]+)`', cells[0])
            assert name, f'command table row without a command: {cells[0]}'
            flags = set(FLAG.findall(cells[1]))
            anchor = re.search(r'\]\(#([^)]+)\)', cells[1])
            if anchor:
                assert anchor.group(1) in heads, f'{cells[0]} points at #{anchor.group(1)}, which no heading names'
                below = [r for s, r in all_tables if s > heads[anchor.group(1)]]
                assert below, f'no table under #{anchor.group(1)}'
                flags |= {f for row in below[0] for f in FLAG.findall(row[0])}
            found[name.group(1)] = flags
    return found


def documenting_row(cmd: str, rows: dict[str, set[str]]) -> str | None:
    """The table row that documents a command: the longest row command that is the command or a prefix of it."""
    hits = [r for r in rows if cmd == r or cmd.startswith(r + ' ')]
    return max(hits, key=len) if hits else None


def undocumented(parsers: dict[str, argparse.ArgumentParser], rows: dict[str, set[str]]) -> list[str]:
    """`command --option` for every option its documenting row does not list (or every option, with no row)."""
    out = []
    for prog, parser in sorted(parsers.items()):
        cmd, flags = command(prog), options(parser)
        row = documenting_row(cmd, rows)
        out += [f'{cmd} {f}' + ('' if row else ' (no row in the command table)')
                for f in sorted(flags - rows.get(row, set()))]
    return out


def strip_external(text: str) -> str:
    """The README without the flags it gives to other tools (uv's own --frozen, --offline, --locked)."""
    return re.sub(r'\buv (?:run|sync)(?: --[a-z][a-z-]*)*', 'uv', text)


def stale(text: str, parsers: dict[str, argparse.ArgumentParser], rows: dict[str, set[str]],
          root: Path = REPO) -> list[str]:
    """Options the README names that the code does not take: anywhere in the text, in a command's row, or in a link
    to a file that does not define them."""
    every = {f for p in parsers.values() for f in options(p)} | {'--help'}
    out = [f'{f} is named but no command takes it' for f in sorted(set(FLAG.findall(strip_external(text))) - every)]
    for row, flags in sorted(rows.items()):
        under = [p for prog, p in parsers.items() if documenting_row(command(prog), rows) == row]
        if not under:
            out.append(f'command table row {row!r} names no command')
        mine = {f for p in under for f in options(p)}
        out += [f'command table row {row!r} lists {f}, which it does not take' for f in sorted(flags - mine)]
    for label, target in re.findall(r'\[`(--[^`]*)`\]\(([^)#?\s]+)', text):
        path = root / target
        if path.is_file():
            body = path.read_text(errors='replace')
            out += [f'[{label}]({target}) links a file that does not define {f}'
                    for f in FLAG.findall(label) if f'"{f}"' not in body and f"'{f}'" not in body and f not in
                    ('--help',)]
    return out


def links(text: str) -> list[str]:
    """Every Markdown link target and HTML href or src."""
    return re.findall(r'\]\(([^)\s]+)\)', text) + re.findall(r'(?:href|src)="([^"]+)"', text)


def broken_links(text: str, root: Path = REPO) -> list[str]:
    """Relative links whose path is not in the tree, anchors that name no heading, and line citations past the end of
    their file. External URLs are left alone."""
    anchors = {slug(h) for _, h in headings(text)}
    out = []
    for link in sorted(set(links(text))):
        if re.match(r'[a-z][a-z0-9+.-]*:', link):
            continue
        path, _, fragment = link.split('?')[0].partition('#')
        if not path:
            if fragment not in anchors:
                out.append(f'#{fragment} names no heading')
            continue
        target = root / path
        if not target.exists():
            out.append(f'{link}: {path} is not in the tree')
            continue
        lines = re.fullmatch(r'L(\d+)(?:-L(\d+))?', fragment or '')
        if lines and target.is_file():
            n = len(target.read_text(errors='replace').splitlines())
            last = max(int(x) for x in lines.groups() if x)
            if last > n:
                out.append(f'{link}: {path} has {n} lines')
    top = {p.name for p in root.iterdir()}
    for path, first, last in re.findall(r'(?<![\w/.-])((?:[\w.-]+/)*[\w-]+\.[a-z]+):(\d+)(?:-(\d+))?\b', text):
        if path.split('/')[0] not in top:
            continue
        target = root / path
        if not target.is_file():
            out.append(f'{path}:{first} cites a file not in the tree')
            continue
        n = len(target.read_text(errors='replace').splitlines())
        if max(int(first), int(last or 0)) > n:
            out.append(f'{path}:{first} is past the end of {path} ({n} lines)')
    return out


@pytest.fixture(scope='module')
def parsers():
    return collect_parsers()


@pytest.fixture(scope='module')
def text():
    return README.read_text()


def test_every_entry_point_is_collected(parsers):
    commands = {command(p) for p in parsers}
    for expected in ('review', 'label', 'board serve', 'board build', 'prepare folder prepare', 'gate score',
                     'compare label', 'checks'):
        assert expected in commands, f'{expected} was not collected from its parser'


def test_every_option_is_in_the_readme_table(parsers, text):
    assert undocumented(parsers, command_table(text)) == []


def test_readme_names_no_option_the_code_lacks(parsers, text):
    assert stale(text, parsers, command_table(text)) == []


def test_repo_links_and_line_citations_resolve(text):
    assert broken_links(text) == []


def test_checks_catch_each_kind_of_staleness(tmp_path):
    """The checks themselves fail on each fault they exist to catch."""
    (tmp_path / 'tool.py').write_text('"--speed"\n' * 3)
    ap = argparse.ArgumentParser(prog='python -m tool')
    ap.add_argument('--speed')
    ap.add_argument('--depth')
    sub = ap.add_subparsers().add_parser('run', prog='python -m tool run')
    sub.add_argument('--fast', action='store_true')
    found = {'python -m tool': ap, 'python -m tool run': sub}
    readme = ('# Usage\n\n| Command | Options |\n|---|---|\n| [`tool`](tool.py) | `--speed`, `--gone` |\n\n'
              'Use [`--depth`](tool.py) or `--never`. See [usage](#usage), [setup](#setup), [tool](tool.py#L9), '
              '[x](missing.py) and tool.py:7.\n')
    rows = command_table(readme)
    assert undocumented(found, rows) == ['tool --depth', 'tool run --fast']
    problems = stale(readme, found, rows, tmp_path)
    assert '--never is named but no command takes it' in problems
    assert "command table row 'tool' lists --gone, which it does not take" in problems
    assert '[--depth](tool.py) links a file that does not define --depth' in problems
    assert broken_links(readme, tmp_path) == ['#setup names no heading', 'missing.py: missing.py is not in the tree',
                                              'tool.py#L9: tool.py has 3 lines',
                                              'tool.py:7 is past the end of tool.py (3 lines)']
