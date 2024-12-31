# Project: Reasoning Uncertainty Calibration

Personal Implementation (not official) of [Forking Paths in Neural Text Generation](https://arxiv.org/abs/2412.07961) using **Llama-3.2-3B-Instruct**.

## Setups

```bash
cd ./src
conda create -n forking_path
conda activate forking_path
pip install -r requirements.txt
```

## Run
**Available Datasets**: Aqua, Coinflip, GSM8K, LastLetters

1. **Base Path Generation**  
   ```bash
   python find_base_path.py --model_name <model_name> 
   ```

2. **Forking Path Generation**  
   ```bash
   python do_resampling.py --model_name <model_name> --base_path_data <generated base path form 1> 
   ```

3. **Bayesian Change Point Detection**  
   Open and run the notebook:
   ```
   BCPD.ipynb
   ```

## TODO
- Add additional baselines including [Semantic Entropy](https://arxiv.org/abs/2302.09664).