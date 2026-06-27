from __future__ import annotations

import argparse
from pathlib import Path

from tools.encoding import configure_utf8_stdio
from workflows.modeling_workflow import ModelingWorkflow, run_from_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the math modeling workflow.")
    parser.add_argument(
        "--problem-file",
        type=Path,
        help="Path to a problem statement file: .txt, .md, .docx or .pdf.",
    )
    parser.add_argument("--problem-text", type=str, help="Problem statement text.")
    parser.add_argument(
        "--data",
        type=Path,
        nargs="*",
        default=[],
        help="Optional data files: csv, tsv, xlsx, xls. If omitted, workspace/data is scanned.",
    )
    parser.add_argument("--use-llm", action="store_true", help="Use optional LLM-backed agents when configured.")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Optional workspace root. Defaults to the project workspace directory.",
    )
    parser.add_argument(
        "--run-workspace",
        action="store_true",
        help="Create an isolated workspace under workspace/runs/ for this workflow run.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Optional isolated run id. Implies --run-workspace and writes to workspace/runs/<run-id>/.",
    )
    parser.add_argument(
        "--export",
        nargs="*",
        choices=["docx", "pdf", "latex"],
        default=[],
        help="Export the paper draft to one or more formats: docx, pdf, latex.",
    )
    return parser


def main() -> None:
    configure_utf8_stdio()

    parser = build_parser()
    args = parser.parse_args()

    if not args.problem_file and not args.problem_text:
        parser.error("Provide --problem-file or --problem-text.")
    if args.workspace and (args.run_workspace or args.run_id):
        parser.error("--workspace cannot be combined with --run-workspace or --run-id.")

    export_formats = args.export or None
    if args.problem_file:
        state = run_from_files(
            args.problem_file,
            args.data,
            use_llm=args.use_llm,
            export_formats=export_formats,
            workspace=args.workspace,
            run_workspace=args.run_workspace,
            run_id=args.run_id,
        )
    else:
        state = ModelingWorkflow(
            use_llm=args.use_llm,
            export_formats=export_formats,
            workspace=args.workspace,
            run_workspace=args.run_workspace,
            run_id=args.run_id,
        ).run(args.problem_text, args.data)

    print("Workflow finished.")
    print(f"Execution status: {state.notes.get('execution_status')}")
    print(f"LLM status: {state.notes.get('llm_status')}")
    print(f"Workspace: {state.workspace.root}")
    for agent_name in ("problem_agent", "modeling_agent", "writing_agent"):
        mode = state.notes.get(f"{agent_name}_mode")
        if mode:
            print(f"{agent_name}: {mode}")
    if state.notes.get("export_formats"):
        print(f"Export formats: {state.notes.get('export_formats')}")
    if state.notes.get("export_errors"):
        print(f"Export errors: {state.notes.get('export_errors')}")
    print(f"Data files: {len(state.data_files)}")
    for path in state.data_files:
        print(f"data: {path}")
    for name, path in state.artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
