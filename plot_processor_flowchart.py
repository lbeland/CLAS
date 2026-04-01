#!/usr/bin/env python3
"""
Render a processor flowchart from a YAML graph definition.

Usage:
    python render_flowchart.py input.yaml -o flowchart.svg
    python render_flowchart.py input.yaml -o flowchart.png
    python render_flowchart.py input.yaml --format pdf

Requirements:
    pip install pyyaml graphviz

Also requires Graphviz installed on the system:
    Ubuntu/Debian: sudo apt install graphviz
    macOS (brew):  brew install graphviz
"""

from __future__ import annotations

import argparse
from itertools import combinations
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from graphviz import Digraph


CONNECTION_RE = re.compile(
    r"""
    ^\s*
    (?P<src_proc>[A-Za-z_]\w*)\.
    (?P<src_port>out|in)\.
    (?P<src_idx>\d+)
    \s*=\s*
    (?P<dst_proc>[A-Za-z_]\w*)\.
    (?P<dst_port>out|in)\.
    (?P<dst_idx>\d+)
    \s*$
    """,
    re.VERBOSE,
)

STATE_REF_RE = re.compile(r"^\s*(?P<proc>[A-Za-z_]\w*)\.(?P<state>[A-Za-z_]\w*)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a processor flowchart from a YAML graph definition."
    )
    parser.add_argument("input", type=Path, help="Input YAML file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path, e.g. flowchart.svg or flowchart.png",
    )
    parser.add_argument(
        "--format",
        choices=["svg", "png", "pdf"],
        default="svg",
        help="Output format if --output is not given explicitly",
    )
    parser.add_argument(
        "--engine",
        choices=["dot", "neato", "fdp", "sfdp", "twopi", "circo"],
        default="dot",
        help="Graphviz layout engine",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Input file not found: {path}")
    except yaml.YAMLError as e:
        raise SystemExit(f"Failed to parse YAML: {e}")

    if not isinstance(data, dict):
        raise SystemExit("Top-level YAML structure must be a mapping/object.")

    if "graph" not in data or not isinstance(data["graph"], dict):
        raise SystemExit("YAML must contain a top-level 'graph' mapping.")

    return data


def parse_connection(connection: str) -> Tuple[str, str, int, str, str, int]:
    """
    Parse strings like:
        Producer.out.0 = Consumer.in.0
    """
    match = CONNECTION_RE.match(connection)
    if not match:
        raise ValueError(f"Invalid connection syntax: {connection!r}")

    src_proc = match.group("src_proc")
    src_port = match.group("src_port")
    src_idx = int(match.group("src_idx"))
    dst_proc = match.group("dst_proc")
    dst_port = match.group("dst_port")
    dst_idx = int(match.group("dst_idx"))

    return src_proc, src_port, src_idx, dst_proc, dst_port, dst_idx


def parse_state_reference(reference: str) -> Tuple[str, str]:
    """
    Parse strings like:
        IAFEstimator.iaf
    """
    match = STATE_REF_RE.match(reference)
    if not match:
        raise ValueError(f"Invalid state reference syntax: {reference!r}")

    return match.group("proc"), match.group("state")


def make_processor_label(name: str, spec: Dict[str, Any]) -> str:
    """
    Build an HTML-like Graphviz label with:
    - processor name
    - class
    - selected options
    """
    proc_class = spec.get("class", "Unknown")
    options = spec.get("options", {}) or {}
    advanced = spec.get("advanced", {}) or {}

    option_lines = []
    for key, value in options.items():
        option_lines.append(f"{key}: {value}")

    advanced_lines = []
    for key, value in advanced.items():
        if isinstance(value, dict):
            advanced_lines.append(f"{key}: {value}")
        else:
            advanced_lines.append(f"{key}: {value}")

    rows = [
        f'<TR><TD BGCOLOR="lightsteelblue"><B>{escape_html(name)}</B></TD></TR>',
        f'<TR><TD ALIGN="LEFT"><B>class:</B> {escape_html(str(proc_class))}</TD></TR>',
    ]

    if option_lines:
        rows.append(
            f'<TR><TD ALIGN="LEFT"><B>options</B><BR ALIGN="LEFT"/>'
            + "<BR ALIGN=\"LEFT\"/>".join(escape_html(line) for line in option_lines)
            + "</TD></TR>"
        )

    if advanced_lines:
        rows.append(
            f'<TR><TD ALIGN="LEFT"><B>advanced</B><BR ALIGN="LEFT"/>'
            + "<BR ALIGN=\"LEFT\"/>".join(escape_html(line) for line in advanced_lines)
            + "</TD></TR>"
        )

    return f"""<
<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="8">
{''.join(rows)}
</TABLE>
>"""


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_graph(data: Dict[str, Any], engine: str = "dot") -> Digraph:
    graph = data["graph"]
    graph_name = graph.get("name", "Flowchart")
    processors = graph.get("processors", {})
    connections = graph.get("connections", [])
    states = graph.get("states", [])

    if not isinstance(processors, dict):
        raise SystemExit("'graph.processors' must be a mapping/object.")

    if not isinstance(connections, list):
        raise SystemExit("'graph.connections' must be a list.")

    if not isinstance(states, list):
        raise SystemExit("'graph.states' must be a list.")

    dot = Digraph(name=graph_name, format="svg", engine=engine)
    dot.attr(rankdir="LR", splines="spline", nodesep="0.6", ranksep="0.9")
    dot.attr(
        "node",
        shape="plaintext",
        fontname="Helvetica",
    )
    dot.attr(
        "edge",
        fontname="Helvetica",
        fontsize="10",
        arrowsize="0.8",
    )
    dot.attr(label=graph_name, labelloc="t", fontsize="20", fontname="Helvetica-Bold")

    # Add processor nodes
    for proc_name, proc_spec in processors.items():
        if not isinstance(proc_spec, dict):
            raise SystemExit(f"Processor '{proc_name}' must be a mapping/object.")
        label = make_processor_label(proc_name, proc_spec)
        dot.node(proc_name, label=label)

    # Add edges from connections
    for raw in connections:
        if not isinstance(raw, str):
            raise SystemExit(f"Each connection must be a string, got: {raw!r}")

        src_proc, src_port, src_idx, dst_proc, dst_port, dst_idx = parse_connection(raw)

        if src_proc not in processors:
            raise SystemExit(f"Connection references unknown source processor: {src_proc}")
        if dst_proc not in processors:
            raise SystemExit(f"Connection references unknown destination processor: {dst_proc}")

        # In the provided format, semantic direction is source = destination
        # Example: Producer.out.0 = Consumer.in.0
        if src_port != "out":
            print(
                f"Warning: source side is not '.out': {raw}",
                file=sys.stderr,
            )
        if dst_port != "in":
            print(
                f"Warning: destination side is not '.in': {raw}",
                file=sys.stderr,
            )

        edge_label = f"out.{src_idx} → in.{dst_idx}"
        dot.edge(src_proc, dst_proc, label=edge_label)

    # Add dashed undirected links between processors that share one or more states.
    shared_state_pairs: Dict[Tuple[str, str], List[str]] = {}
    for state_item in states:
        if not isinstance(state_item, dict) or len(state_item) != 1:
            raise SystemExit(
                "Each element in 'graph.states' must be a mapping with a single state name key."
            )

        state_name, state_spec = next(iter(state_item.items()))
        if not isinstance(state_spec, dict):
            raise SystemExit(f"State '{state_name}' must be a mapping/object.")

        refs = state_spec.get("states", [])
        if not isinstance(refs, list):
            raise SystemExit(f"State '{state_name}.states' must be a list.")

        processors_for_state = set()
        for ref in refs:
            if not isinstance(ref, str):
                raise SystemExit(
                    f"State '{state_name}' references must be strings, got: {ref!r}"
                )
            proc_name, _ = parse_state_reference(ref)
            if proc_name not in processors:
                raise SystemExit(
                    f"State '{state_name}' references unknown processor: {proc_name}"
                )
            processors_for_state.add(proc_name)

        for left, right in combinations(sorted(processors_for_state), 2):
            pair = (left, right)
            shared_state_pairs.setdefault(pair, []).append(str(state_name))

    for (left, right), state_names in shared_state_pairs.items():
        label = "state: " + ", ".join(sorted(set(state_names)))
        dot.edge(
            left,
            right,
            label=label,
            style="dotted",
            color="red3",
            fontcolor="red4",
            dir="both",
            arrowhead="dot",
            arrowtail="dot",
            arrowsize="0.8",
            constraint="false",
        )

    return dot


def infer_output_path(input_path: Path, output: Path | None, fmt: str) -> Tuple[Path, str]:
    if output is not None:
        suffix = output.suffix.lower().lstrip(".")
        if suffix in {"svg", "png", "pdf"}:
            return output.with_suffix(""), suffix
        return output, fmt

    out_base = input_path.with_suffix("")
    return out_base, fmt


def main() -> None:
    args = parse_args()
    data = load_yaml(args.input)
    dot = build_graph(data, engine=args.engine)

    output_base, output_format = infer_output_path(args.input, args.output, args.format)
    dot.format = output_format

    try:
        rendered_path = Path(dot.render(filename=str(output_base), cleanup=True))
    except Exception as e:
        raise SystemExit(
            "Failed to render graph. Make sure Graphviz is installed and available in PATH.\n"
            f"Underlying error: {e}"
        )

    print(f"Flowchart written to: {rendered_path}")


if __name__ == "__main__":
    main()