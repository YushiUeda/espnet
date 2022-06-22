#!/usr/bin/env bash

# Copyright 2022 Yushi Ueda
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# Note: This script is for running the training on 2-spk simulation dataset.
# After running this script, 
# run local/run_adapt_simu.sh (1-4 spk simu data) and 
# local/run_adapt_real.sh (real data (callhome))
# to train and evaluate the callhome dataset.

set -e
set -u
set -o pipefail

train_set="simu/data/swb_sre_tr_ns2_beta2_100000"
valid_set="simu/data/swb_sre_cv_ns2_beta2_500"
test_sets="simu/data/swb_sre_cv_ns2_beta2_500"

train_config="conf/tuning/train_diar_eda_5.yaml"
decode_config="conf/tuning/decode_diar_eda.yaml"

./diar.sh \
    --collar 0.25 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 32 \
    --audio_format flac \
    --local_data_opts "--simu_opts_num_speaker_array 2 --simu_opts_sil_scale_array 2" \
    --hop_length 128 \
    --frame_shift 512 \
    "$@"
