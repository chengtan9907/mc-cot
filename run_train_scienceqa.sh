#!/usr/bin/env bash
# TRAIN - rationale generation
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --model ./models/unifiedqa-t5-base --user_msg rationale --img_type detr --bs 8 --eval_bs 4 --eval_acc 10 --output_len 512 --final_eval --prompt_format QCM-LE --epoch 50 --vot_num 5 --alpha 0.5 --output_dir ./results/base_train_from_scratch
# EVAL - rationale generation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --model ./models/unifiedqa-t5-base --user_msg rationale --final_eval --img_type detr --bs 4 --eval_bs 24 --eval_acc 10 --output_len 512 --prompt_format QCM-LE --evaluate_dir ./results/base_train_from_scratch/rationale
# Train - answer inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --model ./models/unifiedqa-t5-base --user_msg answer --img_type detr --bs 8 --eval_bs 4 --eval_acc 10 --output_len 64 --final_eval --prompt_format QCMG-A --epoch 50 --vot_num 5 --alpha 0.5 --eval_le ./results/base_train_from_scratch/rationale/predictions_ans_eval.json --test_le ./results/base_train_from_scratch/rationale/predictions_ans_test.json --output_dir ./results/base_train_from_scratch
# EVAL - answer inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --model ./models/unifiedqa-t5-base --user_msg answer --img_type detr --bs 4 --eval_bs 24 --eval_acc 10 --output_len 64 --final_eval --prompt_format QCMG-A --eval_le ./results/base_train_from_scratch/rationale/predictions_ans_eval.json --test_le ./results/base_train_from_scratch/rationale/predictions_ans_test.json --evaluate_dir ./results/base_train_from_scratch/answer