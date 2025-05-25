set -e
set -x


TASK="WN18RR"
if [[ $# -ge 1 ]]; then
    TASK=$1
    shift
fi

python preprocess.py \
--task "${TASK}" \
--data-dir "./data/${TASK}/" \
--max-hop-path 5