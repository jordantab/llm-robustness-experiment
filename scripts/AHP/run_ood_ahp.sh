#!/bin/bash

# python ood_ahp_eval.py --model_id mixtral:8x7b --robustness_type ood --benchmark flipkart
# python ood_ahp_eval.py --model_id mixtral:8x7b --robustness_type ood2 --benchmark flipkart
# python scripts/AHP/ood_ahp_eval.py --model_id mixtral:8x7b --robustness_type ood --benchmark ddx
# python scripts/AHP/ood_ahp_eval.py --model_id mixtral:8x7b --robustness_type ood2 --benchmark ddx

# python scripts/AHP/ood_ahp_eval.py --model_id llama2:7b --robustness_type ood --benchmark flipkart
# python ood_ahp_eval.py --model_id llama2:7b --robustness_type ood2 --benchmark flipkart
# python scripts/AHP/ood_ahp_eval.py --model_id llama2:7b --robustness_type ood --benchmark ddx
# python scripts/AHP/ood_ahp_eval.py --model_id llama2:7b --robustness_type ood2 --benchmark ddx

# python scripts/AHP/ood_ahp_eval.py --model_id llama2:13b --robustness_type ood --benchmark flipkart
python scripts/AHP/ood_ahp_eval.py --model_id llama2:13b --robustness_type ood2 --benchmark flipkart
python scripts/AHP/ood_ahp_eval.py --model_id llama2:13b --robustness_type ood --benchmark ddx
python scripts/AHP/ood_ahp_eval.py --model_id llama2:13b --robustness_type ood2 --benchmark ddx