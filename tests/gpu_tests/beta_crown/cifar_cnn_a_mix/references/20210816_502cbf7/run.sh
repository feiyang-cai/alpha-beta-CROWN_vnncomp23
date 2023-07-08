#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: AMD Ryzen 9 5950X + GTX 3090
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

mkdir reference_outputs

python bab_verification_eran.py --load sdp_models/cnn_a_mix.model --model cnn_4layer --data CIFAR_SDP --batch_size 4096 --iteration 20 --timeout 200 --mode verified-acc --start 7 --end 8 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_eran.py --load sdp_models/cnn_a_mix.model --model cnn_4layer --data CIFAR_SDP --batch_size 4096 --iteration 20 --timeout 200 --mode verified-acc --start 18 --end 19 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_eran.py --load sdp_models/cnn_a_mix.model --model cnn_4layer --data CIFAR_SDP --batch_size 4096 --iteration 20 --timeout 200 --mode verified-acc --start 83 --end 84 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_eran.py --load sdp_models/cnn_a_mix.model --model cnn_4layer --data CIFAR_SDP --batch_size 4096 --iteration 20 --timeout 200 --mode verified-acc --start 48 --end 49 --pgd_order skip 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_eran.py --load sdp_models/cnn_a_mix.model --model cnn_4layer --data CIFAR_SDP --batch_size 4096 --iteration 20 --timeout 200 --mode verified-acc --start 164 --end 165 --pgd_order skip 2>&1 | tee reference_outputs/test5_unsafe.out

