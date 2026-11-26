#!/usr/bin/env bash
# Collect TubeTransfer SpaceMouse demos until the dataset holds TARGET successful demos.
#
#   scripts/collect_tubetransfer.sh TARGET
#
#   scripts/collect_tubetransfer.sh 50     # record until 50 demos are saved
#   scripts/collect_tubetransfer.sh 100    # later: top the same dataset up to 100
#
# Each attempt launches play.py --record with the two-camera setup (wrist + side) and a
# fresh random layout, then appends the episode to logs/data/tubetransfer_demos.npy via
# convert_demos.py. Only successful episodes are kept: a failed, timed-out or crashed
# attempt is simply recorded again. After 3 misses in a row it asks whether to go on.
#
# Re-running with the same TARGET resumes where it stopped, because the count comes from
# the dataset itself. Ctrl-C stops cleanly after the current step; the dataset save is
# never interrupted half-way.
#
# Demo style: carry the tube fast and insert it slowly. The policy learns its speeds
# from these demos, and policy-informed motion scaling relies on that contrast.
set -u

TARGET=${1:?usage: $0 TARGET   (total number of demos the dataset should hold)}
case "$TARGET" in ''|*[!0-9]*) echo "TARGET must be a positive integer"; exit 2 ;; esac

cd "$(dirname "$0")/.."
PY=(scripts/isaac_python.sh)

ROLLOUT=logs/rollouts/eval_TubeTransfer_with_SpaceMousePilot
OUTPUT=logs/data/tubetransfer_demos.npy
META=$OUTPUT.tasks.json
VIEWS="wrist:camera_wrist/rgb,side:camera_side/rgb"

count() { python3 -c "import json;print(len(json.load(open('$META'))))" 2>/dev/null || echo 0; }

# Run a command immune to Ctrl-C. In a non-interactive shell, background jobs ignore
# SIGINT, while this script's own trap still records the stop request.
run_protected() {
    "$@" &
    local pid=$! rc
    while :; do
        wait "$pid"; rc=$?
        kill -0 "$pid" 2>/dev/null || break
    done
    return "$rc"
}

stop=0
trap 'stop=1; echo; echo ">>> stop requested - finishing the current step"' INT

finish() {
    echo "Dataset: $(count) demos in $OUTPUT"
    echo "Resume with:  scripts/collect_tubetransfer.sh $TARGET"
}

# ---- pre-flight -------------------------------------------------------------------
if command -v lsusb >/dev/null && ! lsusb -d 256f: >/dev/null 2>&1; then
    echo "SpaceMouse not found (lsusb -d 256f: is empty). Plug it in and re-run."; exit 1
fi
if pgrep -f "scripts/play.py" >/dev/null; then
    echo "Another play.py is running and would hold the SpaceMouse:"; pgrep -af "scripts/play.py"; exit 1
fi

if [ -f "$OUTPUT" ]; then
    backup=logs/data/backups/$(date +%Y%m%d_%H%M%S)
    mkdir -p "$backup" && cp "$OUTPUT" "$META" "$backup"/ && echo "Backed up dataset to $backup/"
fi
echo "Starting at $(count) demos; target $TARGET -> $OUTPUT"
echo "Reminder: carry fast, insert slowly."

# ---- collection loop --------------------------------------------------------------
misses=0
while [ "$(count)" -lt "$TARGET" ]; do
    if [ "$stop" = 1 ]; then finish; exit 130; fi

    before=$(count)
    echo
    echo "================  demo $((before + 1)) / $TARGET   (attempt $((misses + 1)))  ================"

    "${PY[@]}" scripts/play.py --task TubeTransfer --pilot SpaceMousePilot \
        --num_envs 1 --two_cam --record --yes

    # Convert even after a Ctrl-C: it keeps only successful episodes, so this can only
    # save a finished demo, never add a broken one.
    if [ -d "$ROLLOUT" ]; then
        run_protected "${PY[@]}" scripts/convert_demos.py \
            --rollout_dir "$ROLLOUT" --output "$OUTPUT" --append --views "$VIEWS"
    fi

    after=$(count)
    if [ "$after" -gt "$before" ]; then
        echo ">>> saved demo $after / $TARGET"
        misses=0
        continue
    fi

    if [ "$stop" = 1 ]; then finish; exit 130; fi

    misses=$((misses + 1))
    echo ">>> NOT saved (not successful, timed out, or crashed)"
    if [ "$misses" -ge 3 ]; then
        read -r -p ">>> $misses misses in a row. [r]etry or [q]uit? " ans </dev/tty || ans=q
        [ "$stop" = 1 ] && ans=q
        case "$ans" in
            q|Q) finish; exit 1 ;;
            *)   misses=0 ;;
        esac
    else
        echo ">>> recording again"
    fi
done

echo
echo "Done: $(count) demos (target $TARGET)."
finish
