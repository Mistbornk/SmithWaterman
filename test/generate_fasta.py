#!/usr/bin/env python3
"""Generate an isolated, reproducible FASTA pair; never replace existing data."""
import argparse
import pathlib
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", type=int)
    parser.add_argument("identity", type=int, help="exact percentage rounded down")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.length <= 10_000_000 or not 0 <= args.identity <= 100:
        parser.error("length must be 1..10000000; identity must be 0..100")
    rng = random.Random(args.seed)
    ref = [rng.choice("ACGT") for _ in range(args.length)]
    query = ref.copy()
    mutations = args.length - args.length * args.identity // 100
    for index in rng.sample(range(args.length), mutations):
        query[index] = rng.choice([base for base in "ACGT" if base != ref[index]])
    # Existing directories are refused, protecting checked-in fixtures.
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, sequence in (("ref", ref), ("alt", query)):
        with (args.out_dir / (name + ".fasta")).open("x") as output:
            output.write(">" + name + "\n" + "".join(sequence) + "\n")
    print(f"seed={args.seed} length={args.length} mutations={mutations} output={args.out_dir}")


if __name__ == "__main__":
    main()
