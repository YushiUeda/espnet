#!/bin/env bash

cat $1 | sed -e "s/^\(.\+\)\(ns[0-9]\)\(.\+\) .\+/\1\2\3 \2/g" > utt2category