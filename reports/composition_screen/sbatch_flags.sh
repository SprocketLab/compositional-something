#!/bin/bash
# Emit site-specific sbatch flags from cluster_env.sh. Usage:
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) <launcher>.slurm
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster_env.sh"
L="$REPO/reports/composition_screen/logs"
echo "--partition=$PARTITION --gres=$GRES --cpus-per-task=$CPUS --mem=$MEM --output=$L/%x-%j.out --error=$L/%x-%j.err"
