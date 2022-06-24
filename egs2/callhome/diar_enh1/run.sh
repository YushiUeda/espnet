#!/usr/bin/env bash

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

train_set="simu/data/swb_sre_tr_ns2_beta2_100000"
#train_set="simu/data/swb_sre_tr_ns2_beta2_10000"
valid_set="simu/data/swb_sre_cv_ns2_beta2_500"
test_sets="simu/data/swb_sre_cv_ns2_beta2_500"

train_config="conf/tuning/train_diar_enh_convtasnet.yaml"
adapt_config=
decode_config="conf/tuning/decode_diar_enh.yaml"
num_spk=2 # 2, 3

./diar_enh.sh \
    --use_noise_ref true \
    --collar 0.25 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 32 \
    --audio_format wav \
    --spk_num "${num_spk}"\
    --hop_length 64 \
    --frame_shift 64 \
    "$@"
