#!/usr/bin/env bash
set -u
cd <HOME_DIR>/conflict_taxonomy_eval
MODEL=qwen3-8b LAYERS=21 GPU=1 DATASETS=FollowBench GROUP=context_all DIRECTION_MODE=random_split DIR_NPZ=<DATA_DIR>/conflict_taxonomy_eval/dumps/qwen3-8b/NQ-Swap/dev_dedup_p_only/resid_hidden_state__unfiltered_all_pairs.npz DUMP_ROOT=<DATA_DIR>/conflict_taxonomy_eval/dumps STEER_OUT_ROOT=<DATA_DIR>/conflict_taxonomy_eval/results ALPHA_START=0 ALPHA_END=2 ALPHA_STEP=0.5 SKIP_EXISTING=0 bash ./run/run_nqswap_dir_cross_taxonomy.sh 2>&1 | tee <TMP_DIR>/followbench_qwen_random.log
