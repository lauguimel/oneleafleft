#!/bin/bash
# Deploy code + data to Aqua and submit training job
#
# Usage:
#   bash aqua/deploy.sh          # sync code + data + submit
#   bash aqua/deploy.sh --code   # sync code only
#   bash aqua/deploy.sh --data   # sync data only
#   bash aqua/deploy.sh --submit # submit PBS job only

set -e

AQUA="maitreje@aqua.qut.edu.au"
REMOTE_DIR="~/Deforestation"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

sync_code() {
    echo "[1] Syncing code → Aqua..."
    rsync -avz --delete \
        --exclude 'data/' \
        --exclude 'models/' \
        --exclude 'results/' \
        --exclude '.git/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude 'logs/' \
        "$LOCAL_DIR/" "$AQUA:$REMOTE_DIR/"
    echo "    Done."
}

sync_data() {
    echo "[2] Syncing chip data → Aqua..."
    ssh "$AQUA" "mkdir -p $REMOTE_DIR/data/chips"
    rsync -avz --progress \
        "$LOCAL_DIR/data/chips/"*.h5 \
        "$AQUA:$REMOTE_DIR/data/chips/"
    rsync -avz \
        "$LOCAL_DIR/data/chip_index.parquet" \
        "$AQUA:$REMOTE_DIR/data/"
    echo "    Done."
}

submit_job() {
    echo "[3] Submitting PBS job..."
    ssh "$AQUA" "cd $REMOTE_DIR && qsub aqua/train_prithvi.pbs"
}

case "${1:-all}" in
    --code)   sync_code ;;
    --data)   sync_data ;;
    --submit) submit_job ;;
    all|"")   sync_code; sync_data; submit_job ;;
    *)        echo "Usage: $0 [--code|--data|--submit]"; exit 1 ;;
esac
