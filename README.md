# AgentFlow Tool Routing Analysis

This repository is a project built on top of **AgentFlow**.  
The goal of this project is to analyze **how reinforcement learning empowers tool-augmented LLM agents**.

## 🌟 What is AgentFlow?

AgentFlow is a **trainable, tool-integrated agentic framework** designed to overcome the **scalability** and **generalization limits** of today’s tool-augmented reasoning approaches. 

Unlike prevailing approaches such as [Search-R1](https://github.com/PeterGriffinJin/Search-R1) which train a **single LLM** to interleave reasoning steps with tool calls, **AgentFlow** introduces a **modular agentic system** with four specialized modules: 🧭 **Planner**, 🛠 **Executor**, ✅ **Verifier**, and ✍️ **Generator**.

framework_overall

For effective planning and tool use, the framework directly **optimizes planner agent within the system** in an **online fashion** using **Flow-based Group Refined Policy Optimization (Flow-GRPO)**, achieving superior performance across diverse domains with improved tool-calling reliability and long-horizon reasoning capabilities.

flow_grpo

---

## 🔎 Project Overview

We focus on the following diagnostic question:

> **What is the main contribution of the RL training stage in AgentFlow?**

Our hypothesis is that a major contribution of Flow-GRPO is improved **tool selection**.  
In other words, the RL stage can be interpreted as learning a **routing policy over tools**.

In the original AgentFlow system, the planner is responsible for both high-level sub-goal generation and tool selection.  
Our project asks whether the performance gain from Flow-GRPO can be partially explained by better tool routing.

To test this, we replace the planner's tool-selection decision with a lightweight **embedding-based tool selector** and compare it against AgentFlow without GRPO and AgentFlow with GRPO.  
Instead of selecting a tool with the LLM planner, our method embeds the current agent state and available tools into a shared embedding space, then selects the nearest tool prototype.

```text
Original AgentFlow:
State -> LLM Planner selects tool -> Executor -> Verifier -> Memory

Ours:
State -> Embedding Model -> Nearest Tool Prototype -> Executor
```

The method is not intended to fully replace RL.  
Rather, it serves as a diagnostic probe for understanding whether tool selection is a key mechanism behind the RL improvement.

---

## ⚙️ Setup

### Prerequisites

- **Python 3.11** (recommended)

### Installation

```bash
bash setup.sh
source .venv/bin/activate
# (Optional) Install `parallel` for running benchmark experiments in parallel:
sudo apt-get update
sudo apt-get install parallel
```

### Setup Environment Variables

Copy the `.env.template` file from `agentflow/.env.template` and rename it to `.env`, then place it in the `agentflow/` folder. Update the following variables with your own API keys:

- `OPENAI_API_KEY` (for judging reasponse)
- `GOOGLE_API_KEY` (for Google Search tool)
- `DASHSCOPE_API_KEY` ([optional] for calling Qwen-2.5-7B-Instruct as engine for agents and tools)
- `TOGETHER_API_KEY` ([optional] alternative for calling Qwen-2.5-7B-Instruct as engine for agents and tools - recommended for international users)
- More ways: serve Qwen2.5-7B-instruct model with vLLM (details refer to `[serve_vllm_local.md](assets/doc/serve_vllm_local.md)`).

Please check [API Key Setup Guide](assets/doc/api_key.md) for detailed instructions on how to obtain these keys.

```bash
cp agentflow/.env.template agentflow/.env
# Then edit agentflow/.env with your API keys
```

---

## 🧪 Experiments

### Research Question

Flow-GRPO improves AgentFlow, but how exactly?

We test the hypothesis that:

> **The RL stage mainly improves tool-selection ability.**

### Compared Settings


| Setting                          | Description                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| **AgentFlow w/o GRPO**           | Baseline AgentFlow without RL planner training.                     |
| **AgentFlow w/ Embedding, Ours** | Replaces planner-based tool selection with embedding-based routing. |
| **AgentFlow w/ GRPO**            | RL-trained AgentFlow planner using Flow-GRPO.                       |


### Benchmarks


| Benchmark     | Description                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------- |
| **AIME24**    | Mathematical reasoning benchmark requiring precise multi-step reasoning.                    |
| **Bamboogle** | Search-intensive reasoning benchmark requiring information seeking and multi-hop reasoning. |


---

## 📊 Results

### Main Results


| Method                       | AIME24 Acc. (%) | Bamboogle Acc. (%) | Avg. Acc. (%) |
| ---------------------------- | --------------- | ------------------ | ------------- |
| AgentFlow w/o GRPO           | 0.00            | 60.00              | 30.00         |
| AgentFlow w/ Embedding, Ours | 26.67           | 62.00              | 44.34         |
| AgentFlow w/ GRPO            | **33.30**       | **66.00**          | **49.65**     |


The embedding-based selector substantially improves over AgentFlow without GRPO.  
Although it does not fully match GRPO, it recovers a large portion of the gap between the non-RL baseline and the RL-trained model.

### Tool-Use Behavior Analysis

tool_call_ratio_aime24

The non-RL baseline heavily relies on the **Python Code Generator**, suggesting a strong single-tool bias.  
With embedding-based selection, the tool-use distribution becomes more balanced.  
With GRPO, the distribution becomes even more calibrated, with more frequent use of search-oriented tools.

---

## ✅ Takeaways

- The RL stage in AgentFlow can be interpreted as learning a **routing policy over tools**.
- A simple embedding-based selector recovers a substantial portion of the GRPO gain without RL training.
- Tool selection appears to be one key mechanism behind AgentFlow's improvement.
- RL likely improves more than routing alone, including sub-goal generation, stopping decisions, memory usage, and recovery from intermediate mistakes.
- For LLM agents, the main challenge of RL may not be optimization itself, but effective and efficient exploration over tool-use trajectories.

---

## 📚 Citation

```bibtex
@inproceedings{li2026flow,
    title = {In-the-Flow Agentic System Optimization for Effective Planning and Tool Use},
    author = {Li, Zhuofeng and Zhang, Haoxiang and Han, Seungju and Liu, Sheng and Xie, Jianwen and Zhang, Yu and Choi, Yejin and Zou, James and Lu, Pan},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2026}
}
```