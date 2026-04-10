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

submit_train() {
    echo "[3] Submitting training job..."
    ssh "$AQUA" "cd $REMOTE_DIR && qsub aqua/train_prithvi.pbs"
}

submit_predict() {
    local YEAR="${2:-2026}"
    echo "[3] Submitting prediction job (year=$YEAR)..."
    ssh "$AQUA" "cd $REMOTE_DIR && qsub -v PRED_YEAR=$YEAR aqua/predict.pbs"
}

case "${1:-all}" in
    --code)    sync_code ;;
    --data)    sync_data ;;
    --train)   submit_train ;;
    --predict) submit_predict "$@" ;;
    all|"")    sync_code; sync_data; submit_train ;;
    *)
        echo "Usage: $0 [--code|--data|--train|--predict YEAR]"
        echo ""
        echo "  (no args)     Sync code + data + submit training"
        echo "  --code        Sync code only"
        echo "  --data        Sync chip HDF5 files only"
        echo "  --train       Submit training PBS job"
        echo "  --predict Y   Submit inference PBS job for year Y (default: 2026)"
        exit 1
        ;;
esac
