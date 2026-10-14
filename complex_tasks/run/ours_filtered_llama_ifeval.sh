#!/usr/bin/env bash
set -euo pipefail
source <DATA_DIR>/.venv/bin/activate
cd <HOME_DIR>/conflict_taxonomy_eval
export MODEL=llama3.1-8b-it LAYERS=10 GPU=5 BATCH_SIZE=4 DATASETS=IFEval GROUP=context_all
export DIR_NPZ=<HOME_DIR>/rerun/dumps/llama3.1-8b-it/NQ-Swap/dev_dedup_by_question_only_p_no_table_li/resid_hidden_state__stage1_correct_use_wrong_context.npz
export DIR_TAG=Ours__stage1_correct_use_wrong_context DUMP_ROOT=<DATA_DIR>/conflict_taxonomy_eval/dumps
export STEER_OUT_ROOT=<HOME_DIR>/rerun/controls/transfer/ours_filtered_533
export ALPHA_START=0 ALPHA_END=2 ALPHA_STEP=0.5 MAX_SAMPLES=500 SKIP_EXISTING=0
bash ./run/run_nqswap_dir_cross_taxonomy.sh
