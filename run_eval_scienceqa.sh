#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py --model ./models/unifiedqa-t5-base --img_type detr --bs 4 --eval_bs 24 --eval_acc 10 --output_len 64 --final_eval --prompt_format QCMG-A --epoch 20 --eval_le ./results/base_pretrained_scienceqa/rationale/predictions_ans_eval.json --test_le ./results/base_pretrained_scienceqa/rationale/predictions_ans_test.json --evaluate_dir ./results/base_pretrained_scienceqa/answer