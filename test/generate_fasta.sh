#!/usr/bin/env bash

# Usage: ./make_fasta.sh <length> <similarity_percent>
# Example: ./make_fasta.sh 1000 90

if [ $# -ne 2 ]; then
  echo "Usage: $0 <length> <similarity_percent>"
  exit 1
fi

LEN=$1
SIM=$2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
OUTDIR="${SCRIPT_DIR}/data"   # ==> <repo>/test/data
mkdir -p "$OUTDIR"

if ! [[ "$LEN" =~ ^[0-9]+$ ]] || ! [[ "$SIM" =~ ^[0-9]+$ ]]; then
  echo "Error: length and similarity must be integers"
  exit 1
fi

if [ "$SIM" -gt 100 ] || [ "$SIM" -lt 0 ]; then
  echo "Error: similarity must be 0–100"
  exit 1
fi

# nucleotides
NUC=(A C G T)

# ----------------------
# generate reference seq
# ----------------------
REF=""
for ((i=0; i<$LEN; i++)); do
  REF+=${NUC[$RANDOM % 4]}
done

# write ref fasta
echo ">ref" > "$OUTDIR/ref.fasta"
echo "$REF" >> "$OUTDIR/ref.fasta"


# ----------------------
# generate alt sequence
# ----------------------

# number of identical bases
IDENTICAL=$(( LEN * SIM / 100 ))
MUTATE=$(( LEN - IDENTICAL ))

ALT=("$REF")   # start identical to ref

# randomly pick positions to mutate
POS_LIST=($(shuf -i 0-$((LEN-1)) -n $MUTATE))

for POS in "${POS_LIST[@]}"; do
  orig=${REF:$POS:1}

  # choose a nucleotide different from original
  new=$orig
  while [ "$new" == "$orig" ]; do
    new=${NUC[$RANDOM % 4]}
  done

  ALT="${ALT:0:$POS}$new${ALT:$((POS+1))}"
done

# write alt fasta
echo ">alt" > "$OUTDIR/alt.fasta"
echo "$ALT" >> "$OUTDIR/alt.fasta"

echo "Done! Generated ref.fasta and alt.fasta"
echo "Length: $LEN, Similarity: $SIM%, Differences: $MUTATE"
