#!/bin/bash

python ood_ahp_eval.py --model_id mixtral:8x7b --robustness_type ood --benchmark flipkart
# python ood_ahp_eval.py --model_id llama2:7b --robustness_type ood --benchmark flipkart
python ood_ahp_eval.py --model_id mixtral:8x7b --robustness_type ood2 --benchmark flipkart
# python ood_ahp_eval.py --model_id llama2:7b --robustness_type ood2 --benchmark flipkart