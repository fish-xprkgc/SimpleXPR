set -e
set -x


TASK="FB15k237"
if [[ $# -ge 1 ]]; then
    TASK=$1
    shift
fi

python preprocess.py \
--task "${TASK}" \
--data-dir "./data/${TASK}/" \
--max-hop-path 2