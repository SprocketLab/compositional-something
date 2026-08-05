#!/bin/bash
# Emit site-specific sbatch overrides. Usage:
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) <script>_portable.slurm
# Command-line flags override any #SBATCH headers in the script.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"
LOGS=$REPO/reports/composition_screen/logs
mkdir -p "$LOGS"
echo "--partition=$PARTITION --gres=$GRES --output=$LOGS/%x-%j.out --error=$LOGS/%x-%j.err"
