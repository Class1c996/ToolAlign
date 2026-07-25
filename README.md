# ToolAlign：基于可执行反馈的工具调用 Agent 后训练系统

ToolAlign 是一个面向 LLM Post-Training、Agent Alignment 与 Tool-Use Alignment 的可复现研究项目。项目不把目标简化为“选一个工具”，而是训练并评估 Agent 在确定性环境中完成完整闭环：判断是否需要工具、选择正确工具、生成合法且正确的参数、读取结果、进行多步调用、在缺参时澄清、达到目标后停止并给出真实回答。

## 研究问题

1. 在同一基座模型上，Function Calling SFT 能否让模型稳定输出可执行调用？
2. 引入可执行环境与 GRPO 后，终态任务成功率是否继续提升，是否减少冗余调用、虚假成功和幻觉工具？
3. terminal-only reward 与 shaped reward 的收益和 reward hacking 风险分别是什么？
4. 在 Seen/Unseen tools、无关请求、缺参澄清和错误参数场景下，模型能否泛化？
5. 在 RTX 5080 16GB 上，如何用 QLoRA 控制显存并获得可复现实验结果？

## 核心流程

```text
任务 + 工具 schema
      ↓
Base prompting → Function Calling SFT → 可执行环境 rollout → GRPO
      ↓                         ↓
JSON/SQLite 状态、确定性工具执行、逐步轨迹记录
      ↓
奖励：终态成功为主，格式/工具/参数为辅助，非法和冗余行为受罚
      ↓
BFCL + 自建任务集评测 → 日志、表格、图表、Demo、技术报告
```

## 系统组成

- `models/`：基座模型与 LoRA 的加载逻辑；模型权重需单独下载，不纳入仓库。
- `envs/`：本地订单、日程、旅行等确定性工具环境，JSON/SQLite 状态和工具执行器。
- `data/`：数据构建逻辑、环境清单，以及可直接运行评测的 seen、unseen、challenge/holdout 合成测试集；原始数据和训练集不纳入仓库。
- `training/`：SFT 与 GRPO 训练入口、数据 collator、checkpoint 与恢复逻辑。
- `rewards/`：终态、格式、工具选择、参数、冗余、幻觉和虚假成功奖励/惩罚。
- `eval/`：BFCL（至少 non-live）适配、自建确定性评测、Seen/Unseen 分组和指标计算。
- `scripts/`：数据准备、训练、评测、绘图和一键复现实验脚本。
- `configs/`：模型、数据、环境、训练和奖励的 YAML 配置。
- `demo/`：命令行或 Gradio 演示。
- `checkpoints/`：本地 LoRA adapters 和训练状态，不纳入仓库。
- `reports/`：保留小型汇总指标；逐条 rollout、评测明细和运行日志不纳入仓库。

研究计划、实验记录和项目审阅材料保存在本地，不随公开代码仓库发布。

## 快速开始

> 下面是开发完成后的约定命令；所有随机种子、数据版本和配置都应写入日志。

```powershell
cd ToolAlign
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\build_env.py --config configs\env.yaml
python scripts\prepare_data.py --config configs\data.yaml
python scripts\train_sft.py --config configs\sft_qwen3_1p7b.yaml
python scripts\train_grpo.py --config configs\grpo_terminal.yaml
python scripts\evaluate.py --config configs\eval.yaml --checkpoint checkpoints\sft
python demo\app.py --checkpoint checkpoints\grpo_shaped
```

当前无需 GPU 即可验证本地闭环：

```powershell
.venv\Scripts\python.exe scripts\build_env.py
.venv\Scripts\python.exe scripts\prepare_data.py --count 1200
.venv\Scripts\python.exe scripts\validate_tasks.py
.venv\Scripts\python.exe scripts\run_rollout.py --policy gold
.venv\Scripts\python.exe scripts\evaluate_local.py
```

也可以直接运行：

```powershell
.venv\Scripts\python.exe scripts\reproduce.py
```

公开数据入口为 `Salesforce/xlam-function-calling-60k`，使用 `scripts\prepare_xlam.py` 下载、去重并保留原始记录；该步骤需要安装 `datasets`，不影响本地自建任务闭环。

### W0 环境 smoke test

W0 的最小验证不依赖第三方包或 GPU。若系统没有 `python` 命令，可使用项目运行时提供的 Python 可执行文件替换下面命令中的 `python`。

```powershell
python scripts\smoke_test.py
```

预期输出为一行 JSON，包含 `"status": "PASS"`、稳定的 `replay_digest` 和 `"tool_count": 2`。测试覆盖白名单工具执行、状态变更、缺少必填参数、未知工具和固定 seed 下的 replay 一致性。

本地硬件基线是 RTX 5080 16GB：主实验使用 Qwen3-1.7B 的 4bit QLoRA，0.6B 只用于冒烟测试；4B 作为 Colab/A100 扩展，不能假设本地稳定训练 8B。训练前先用 `scripts\smoke_test.py` 验证单 batch、工具执行和奖励计算。

## 最终展示方式

最终 Demo 输入自然语言任务，实时展示 Agent 的每一步：当前状态、工具名、JSON 参数、工具结果、奖励分解和最终回答。报告用同一版本配置给出 Base、SFT、SFT+GRPO terminal-only、SFT+GRPO shaped reward 四组结果，并报告格式合法率、工具选择准确率、参数 F1、端到端成功率、幻觉工具率、虚假成功率、平均调用次数和延迟。

## 仓库内容说明

为控制仓库体积并避免发布实验过程材料，本仓库包含复现所需的源码、配置、依赖说明、小型汇总指标，以及 `data/processed/` 下的合成测试集。模型权重、训练检查点、原始数据、训练集、运行日志、逐条模型评测结果及内部 Markdown 文档均通过 `.gitignore` 排除。
