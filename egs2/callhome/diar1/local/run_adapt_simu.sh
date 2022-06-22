#!/usr/bin/env bash

# Copyright 2022 Yushi Ueda
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# Note: This script is used for adapting to 1-4 spk simu data using the model
# pretrained on 2 spk simu data (run.sh).
# local/run_adapt_real.sh should be followed after running this script.

set -e
set -u
set -o pipefail

train_set="simu/data/swb_sre_tr_ns1n2n3n4_beta2n2n5n9_100000"
valid_set="simu/data/swb_sre_cv_ns1n2n3n4_beta2n2n5n9_500"
test_sets="simu/data/swb_sre_cv_ns1n2n3n4_beta2n2n5n9_500"

train_config="conf/tuning/train_diar_eda_adapt.yaml"
decode_config="conf/tuning/decode_diar_eda.yaml"
# change the path according to the actual path to the pretrained model
pretrained="exp/diar_train_diar_eda_5_raw/latest.pth"

./diar.sh \
    --collar 0.25 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --local_data_opts "--stage 1 --stop_stage 1" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 32 \
    --audio_format flac \
    --num_spk 4 \
    --diar_args "--init_param ${pretrained}" \
    --diar_tag "train_diar_eda_adapt_simu" \
    --hop_length 128 \
    --frame_shift 512 \
    "$@"