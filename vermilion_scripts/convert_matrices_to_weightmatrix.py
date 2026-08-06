#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
from decimal import Decimal, ROUND_FLOOR, getcontext
from pathlib import Path


getcontext().prec = 50
DEFAULT_ROW_SUM = 32768


def parse_matrix(path: Path) -> list[list[Decimal]]:
    rows: list[list[Decimal]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append([Decimal(token) for token in line.split()])

    if not rows:
        raise ValueError(f"{path} is empty")

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{path} is not rectangular")

    return rows


def normalize_row_to_sum(row: list[Decimal], target_sum: int) -> list[int]:
    nonneg = [max(value, Decimal(0)) for value in row]
    row_sum = sum(nonneg)

    if row_sum == 0:
        base = [target_sum // len(nonneg)] * len(nonneg)
        for idx in range(target_sum % len(nonneg)):
            base[idx] += 1
        return base

    scaled = [value * Decimal(target_sum) / row_sum for value in nonneg]
    floored = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in scaled]
    remainder = target_sum - sum(floored)

    ranked = sorted(
        ((scaled[idx] - Decimal(floored[idx]), idx) for idx in range(len(scaled))),
        reverse=True,
    )
    for _, idx in ranked[:remainder]:
        floored[idx] += 1

    if sum(floored) != target_sum:
        raise ValueError("normalized row does not hit target sum")

    return floored


def convert_matrix(path: Path, target_sum: int) -> list[list[int]]:
    matrix = parse_matrix(path)
    return [normalize_row_to_sum(row, target_sum) for row in matrix]


def write_weight_matrix(path: Path, matrix: list[list[int]]) -> None:
    payload = "[" + ", ".join("[" + ", ".join(str(value) for value in row) + "]" for row in matrix) + "]"
    path.write_text(payload + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert matrices/*.txt into ffapp-compatible weight matrices."
    )
    parser.add_argument(
        "--input-dir",
        default="matrices",
        help="Directory containing whitespace-separated source matrices.",
    )
    parser.add_argument(
        "--output-dir",
        default="../test",
        help="Directory for generated weight-matrix files.",
    )
    parser.add_argument(
        "--row-sum",
        type=int,
        default=DEFAULT_ROW_SUM,
        help="Exact integer row sum required by ffapp/load_weight_matrix().",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Validate that generated files parse as list-of-lists with the requested row sum.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    created: list[Path] = []
    for src in sorted(input_dir.glob("*.txt")):
        converted = convert_matrix(src, args.row_sum)
        dst = output_dir / src.name
        write_weight_matrix(dst, converted)
        created.append(dst)

    for dst in created:
        print(dst)

    if args.check_existing:
        for dst in created:
            parsed = ast.literal_eval(dst.read_text().strip())
            for row in parsed:
                if sum(row) != args.row_sum:
                    raise SystemExit(f"{dst} has row sum {sum(row)} instead of {args.row_sum}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
