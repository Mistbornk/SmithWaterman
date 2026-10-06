#!/bin/bash

# Number of runs
N=100

success=0
fail=0

for i in $(seq 1 $N); do
    echo "===== RUN $i/$N ====="

    # Generate test data
    ./generate_fasta.sh 10000 70
    if [ $? -ne 0 ]; then
        echo "Error: generate_fasta.sh failed!"
        exit 1
    fi

    # Run the test
    ../build/test/biovoltron-test
    exitcode=$?

    if [ $exitcode -eq 0 ]; then
        echo "[PASS] Run $i"
        success=$((success + 1))
    else
        echo "[FAIL] Run $i"
        fail=$((fail + 1))
    fi
done

echo
echo "====================== SUMMARY ======================"
echo "Total runs: $N"
echo "PASS:  $success"
echo "FAIL:  $fail"

if [ $N -gt 0 ]; then
    rate=$(echo "scale=4; $success / $N" | bc)
    echo "Accuracy: $rate"
fi

echo "======================================================"
