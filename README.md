# Style over Substance: Failure modes of LLM judges in alignment benchmarking

![](./figures/mismo-fig.png)

This codebase stores the complete artifacts and describes how to reproduce or extend the results from the paper "Style over Substance: Failure modes of LLM judges in alignment benchmarking".

# List of Benchmarks

In this table, you can find the complete list of benchmarks we use in **SOS-Bench**, along with the codebase necessary to run them. Below, we describe how to work with each codebase.

| **Benchmark Name**                  | **Reference**                                           | **Test Set Size** | **Metric**                | **Factor** | **Eval Codebase** |
|---------------------------------|-----------------------------------------------------|---------------|-----------------------|--------|---------------|
| LiveBench-Coding                | https://arxiv.org/abs/2406.19314                    | 130           | Exact Match Acc       | WK     | LiveBench     |
| LiveBench-Data Analysis         | https://arxiv.org/abs/2406.19314                    | 150           | Exact Match Acc       | WK     | LiveBench     |
| LiveBench-Instruction Following | https://arxiv.org/abs/2406.19314                    | 200           | Exact Match Acc       | IF     | LiveBench     |
| LiveBench-Language              | https://arxiv.org/abs/2406.19314                    | 140           | Exact Match Acc       | WK     | LiveBench     |
| LiveBench-Math                  | https://arxiv.org/abs/2406.19314                    | 230           | Exact Match Acc       | WK     | LiveBench     |
| LiveBench-Reasoning             | https://arxiv.org/abs/2406.19314                    | 150           | Exact Match Acc       | WK     | LiveBench     |
| IFEval                          | https://arxiv.org/abs/2311.07911                    | 540           | Avg of Custom Metrics | IF     | Eleuther      |
| MATH Lvl 5                      | https://arxiv.org/abs/2103.03874                    | 1000          | Exact Match Acc       | WK     | Eleuther      |
| MuSR                            | https://arxiv.org/abs/2310.16049                    | 750           | Acc                   | WK     | Eleuther      |
| GPQA                            | https://arxiv.org/abs/2311.12022                    | 1250          | Acc                   | WK     | Eleuther      |
| MMLU-Pro                        | https://arxiv.org/abs/2406.01574                    | 12000         | Acc                   | WK     | Eleuther      |
| BBH                             | https://arxiv.org/abs/2210.09261                    | 6750          | Acc                   | WK     | Eleuther      |
| BeaverTails                     | https://arxiv.org/abs/2307.04657                    | 1400          | Acc                   | Safety | Eleuther      |
| CDNA                            | https://huggingface.co/datasets/walledai/CDNA       | 2730          | Acc                   | Safety | Eleuther      |
| DTToxicity                      | https://huggingface.co/datasets/walledai/DTToxicity | 4800          | Acc                   | Safety | Eleuther      |
| JailbreakHub                    | https://arxiv.org/abs/2308.03825                    | 15100         | Acc                   | Safety | Eleuther      |
| BBQ                             | https://arxiv.org/abs/2110.08193                    | 58500         | Acc                   | Safety | Eleuther      |
| WMDP                            | https://arxiv.org/abs/2403.03218                    | 3670          | Inverse Acc           | Safety | Eleuther      |
| XSTest                          | https://arxiv.org/abs/2308.01263                    | 450           | Acc                   | Safety | Eleuther      |
| WildGuardTest                   | https://arxiv.org/abs/2406.18495                    | 1730          | Acc                   | Safety | Eleuther      |
| Toxigen                         | https://arxiv.org/abs/2203.09509                    | 9900          | Acc                   | Safety | Eleuther      |
| StrongREJECT                    | https://arxiv.org/abs/2402.10260                    | 310           | Acc                   | Safety | Eleuther      |
| SGXSTest                        | https://huggingface.co/datasets/walledai/SGXSTest   | 100           | Acc                   | Safety | Eleuther      |
| SaladBench                      | https://arxiv.org/abs/2402.05044                    | 30400         | Acc                   | Safety | Eleuther      |

# List of Artifacts in this Repository

Here is a brief description of our result artifacts.

## Eleuther Results

```
Filenames: eleuther_wandb.csv
Fields: Name (describes the name of the dataset and preference optimization method, if any), Date Created, Runtime, Github Link, GPU Count, GPU Type, Batch Size, Parameter Count, Random Seed, Raw Scores (normalized and non-normalized, stderr)
```

## Arena-Hard-Auto Results

```
Filenames: arena_hard_auto.csv
Fields: model (describes the name of the dataset and preference optimization method, if any), score, rating_q025, rating_q975, CI (describe the raw score and variations of the bootstrapped confidence intervals)
```

## LiveBench Results

```
Filenames: livebench_groups.csv, livebench_tasks.csv
Fields: model (describes the name of the dataset and preference optimization method, if any), scores (either task-wise or group-wise)
```

# How to Run SOS-Bench

The entirety of SOS-Bench can be run as a two-stage process; the first set of benchmarks can be completed using a fork of the Eleuther AI Harness, and the second set can be run using the LiveBench codebase.

## Eleuther

1. `pip install lm_eval[wandb,vllm,math,ifeval], sentencepiece`
2. `python install_nltk_punkt.py`
3. Git clone our [Eleuther AI Harness fork](https://anonymous.4open.science/r/lm-evaluation-harness-24C3/README.md) which contains additional tasks
4. `cd lm-evaluation-harness`
5. `pip install -e .`
6. `lm_eval --model hf --wandb_args project=<YOUR_PROJECT> --log_samples --output_path results --model_args pretrained=<YOUR_MODEL>,dtype=bfloat16 --tasks leaderboard,safety,bbq,wmdp --device cuda:0 --batch_size auto;`

## LiveBench

1. Git clone the [LiveBench repository](https://github.com/livebench/livebench)
2. Follow the instructions provided in the repository readme.

# Citation
