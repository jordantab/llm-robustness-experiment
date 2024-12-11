# 11785 Team 20 Project: On Adversarial Robustness and Out-of-Distribution Robustness of Large Language Models

This repository contains the code we developed to evaluate the correlation between the adversarial and out-of-distribution robustness of several LLMs. The experiment details can be found [here](https://drive.google.com/file/d/1BTr7b6THeWSonS3ljpzYTznAFCzEp4gs/view?usp=sharing).


## Project Setup
We conducted the evaluation using Ollama and Nvidia T4/GH200 GPUs. Before getting started, you will need access to compute capable of repetitive LLM inference using llama2:7b, llama2:13b, and mixtral:8x7b


1. **Set Up Environment:**
   Create a virtual environment:
   ```bash
   python -m venv pb_env
   source pb_env/bin/activate
   ```
2. **Install Dependencies:**
   Make sure you have Python 3.10+ installed. You can install the required packages using pip:   
   ```bash
   pip install -r requirements.txt   
   ```
3. **Install Ollama:**
    Follow the instructions [here](https://ollama.com/download/linux) to install Ollama
4. **Download Models with Ollama:**
   You will need to download the LLM models with Ollama. After you install Ollama, you can download the models by running the following commands:
   ```bash
   ollama run llama2:7b
   ollama run llama2:13b
   ollama run mixtral:8x7b
   ```

## Running the Evaluation

### Baseline Evaluation

The scripts for the baseline evaluation can be found in the `scripts/Baseline` directory. 
- PromptRobust: `pb_eval.py`, to run the evaluation, you can run the following command:
  ```bash
  python scripts/Baseline/pb_eval.py
  ```
- AdvGLUE++: `advglueplusplus_eval.py`, to run the evaluation, you can run the following command:
  ```bash
  bash scripts/Baseline/run_ag.sh
  ```
- Flipkart: `flipkart_eval.py`, to run the evaluation, you can run the following command:
  ```bash
  python scripts/Baseline/flipkart_eval.py
  ```

### AHP Evaluation

The scripts for the AHP evaluation can be found in the `scripts/AHP` directory.
- Running AHP for Adversarial Robustness: 
    ```bash
    bash scripts/AHP/run_adv_ahp.sh
    ```
- Running AHP for Out-of-Distribution Robustness: 
    ```bash
    bash scripts/AHP/run_ood_ahp.sh
    ```

### ICR Evaluation

The scripts for the ICR evaluation can be found in the `scripts/ICR` directory.



