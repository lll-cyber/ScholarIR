# ScholarIR

赛题三：复杂学术查询的智能论文搜索。

## 四阶段流水线

| # | 包名 | 对应能力 | 状态 |
|---|------|----------|------|
| (1) | `query_understanding/` | 查询理解与分解（意图、实体、子查询、改写扩展） | **已实现** |
| (2) | `search/` + `filter/` | 自主搜索（含 broaden/narrow）+ 过滤/LLM 精判/impact 融合 | **已实现** |
| (3) | `ranking/` | 按 Filter 分数排序 + threshold / arxiv_only / max_return 截断 | **已实现**（薄层） |
| (4) | `organize/` | 分档列表 + 引用关系图 + 自然语言入选理由 | **已实现** |

编排：`scholar_ir.run(question)` → `pipeline.py`

接口约定见：`../技术路线/核心模块IO约定.md`

## 目录

```text
ScholarIR/
├── configs/
├── data/pasa-dataset -> …
├── logs/
├── outputs/
├── scripts/
├── src/scholar_ir/
│   ├── types.py
│   ├── config.py
│   ├── pipeline.py              # 四阶段编排
│   ├── query_understanding/     # (1) 查询理解与分解
│   ├── search/                  # (2) 自主搜索（多源 adapt + 迭代）
│   │   └── adapt/               # arxiv / openalex / semantic
│   ├── filter/                  # (2) 候选过滤（关键词 + LLM + impact）
│   ├── ranking/                 # (3) 排序截断（分数由 filter 产出）
│   ├── organize/                # (4) 列表 / 关系图 / 入选理由
│   ├── eval/                    # 集合 P/R/F1
│   ├── llm/
│   └── vendor_spar/
└── tests/
```

## 快速开始

```bash
cd /data/coding/ScholarIR
export PYTHONPATH=/data/coding/ScholarIR/src:$PYTHONPATH

# (1) 查询理解
python3 scripts/demo_understanding.py --deepseek

# (2) 搜索 adapt
python3 scripts/demo_retrieval_arxiv.py --live
python3 scripts/test_s2_api.py              # Semantic Scholar (S2_API_KEY)
python3 scripts/demo_retrieval_semantic.py --live

# 全链路 smoke（落盘中间结果）
python3 scripts/smoke_pipeline.py --deepseek --out-dir outputs/smoke_run

# PaSa 集合 F1
python3 scripts/eval_pasa.py --split auto --limit 5 --deepseek
```

`pipeline` 选项键（新旧均可）：

| 阶段 | 新键 | 兼容旧键 |
|------|------|----------|
| (1) | `query_understanding` | `understanding` |
| (2) 搜索 | `search` | `retrieval` |
| (2) 过滤 | `filter` | `judge` |
| (3) | `ranking` | — |
| (4) | `organize` | — |

## 当前状态

- (1) Understanding：slots + terms + Slot Usage；DeepSeek 可选；`add_survey_modifier` 默认关
- (2) Search：默认 **arxiv + openalex + semantic** 三源（`DEFAULT_SOURCES`）
- (2) Filter：硬规则（年/否定）+ 关键词 + 可选 LLM；`relevance_criteria` 进 LLM 软判定（非硬淘汰）；impact 融合（引用/时效/venue/标题密度，**不是**向量 embedding）+ 跨 intent 归一化
- (3) Ranking：不再重算分 / 不做 embedding；只做排序与截断（模块名仍叫 ranking；职责≈select/truncate）。`arxiv_only` 默认 False，评测/smoke 请显式 True
- (4) Organize：高度/部分相关分档、`selection_reason`、引用图（OpenAlex / Crossref / S2 回退）
- 日志：`run(..., {"log_file": "logs/run.log"})` 或环境变量 `SCHOLAR_IR_LOG_FILE`；JSON 落盘仍用 `scripts/smoke_pipeline.py --out-dir`
- `embeddings.py`：独立向量客户端，**尚未接入** filter/ranking（与 impact 特征不同）
