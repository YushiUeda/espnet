#!/usr/bin/env bash

# Copyright 2022 Yushi Ueda
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# Note: This script is used for adapting to real data using the model
# pretrained on 1-4 spk simu data (run_adapt_simu.sh).

set -e
set -u
set -o pipefail

train_set="callhome1_spkall"
valid_set="callhome2_spkall"
test_sets="callhome2_spkall"

train_config="conf/tuning/train_diar_eda_adapt.yaml"
decode_config="conf/tuning/decode_diar_eda.yaml"
# change the path according to the actual path to the pretrained model
pretrained="exp/diar_train_diar_eda_adapt_simu/latest.pth"

./diar.sh \
    --stage 2 \
    --collar 0.25 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 32 \
    --audio_format flac \
    --num_spk 7 \
    --diar_args "--init_param ${pretrained} --optim_conf lr=0.001 --attractor_conf attractor_grad=False" \
    --diar_tag "train_diar_eda_adapt_real_lr0001" \
    --hop_length 128 \
    --frame_shift 512 \
    "$@"