# ScholarIR

赛题三：复杂学术查询的智能论文搜索。

## 四阶段流水线

| # | 包名 | 对应能力 | 状态 |
|---|------|----------|------|
| (1) | `query_understanding/` | 查询理解与分解（意图、实体、子查询、改写扩展） | **已实现** |
| (2) | `search/` + `filter/` | 自主搜索 + 过滤不相干/低质量；迭代策略后续补 | 搜索已实现；过滤为 stub |
| (3) | `ranking/` | 论文综合排序（细粒度相关性） | **骨架 stub** |
| (4) | `organize/` | 搜索结果归纳整理（列表/关系图等） | **骨架 stub** |

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
│   ├── search/                  # (2) 自主搜索（多源 adapt）
│   │   └── adapt/               # arxiv / openalex / semantic
│   ├── filter/                  # (2) 候选过滤（原 judge，stub）
│   ├── ranking/                 # (3) 综合排序（stub）
│   ├── organize/                # (4) 结果归纳整理（stub）
│   ├── eval/                    # 集合 P/R/F1
│   ├── llm/
│   └── vendor_spar/
└── tests/
```

## 快速开始

```bash
cd /data3/ai_inn/ScholarIR
export PYTHONPATH=/data3/ai_inn/ScholarIR/src:$PYTHONPATH

# (1) 查询理解
python3 scripts/demo_understanding.py --deepseek

# (2) 搜索 adapt
python3 scripts/demo_retrieval_arxiv.py --live
python3 scripts/test_s2_api.py              # Semantic Scholar (S2_API_KEY, 1 req/s)
python3 scripts/demo_retrieval_semantic.py --live

# 全链路 smoke
python3 scripts/smoke_pipeline.py

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

- (1) Understanding：slots + terms + Slot Usage；DeepSeek 可选
- (2) Search：默认 **arxiv + openalex + semantic (S2)** 三源 union；S2 全局限速 `S2_RATE_LIMIT_RPS=1`
- (2) Filter：stub pass-all（原 Judge）
- (3)(4)：透传骨架，待实现
