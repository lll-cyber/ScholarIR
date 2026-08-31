# PaperSeek 设计要点 — 可供 ScholarIR 借鉴

> 基于 `/data/coding/paperseek` 代码梳理，排除「本地 Embedding 模型」方案后，整理出以下可直接参考的设计点。

---

## 1. 意图驱动的查询生成

PaperSeek 在生成任何数据库查询前，先用 LLM 做一次**意图分析**，输出稳定的结构化约束：

- `intent`：研究意图
- `core_concepts`：核心概念
- `likely_synonyms`：可能的同义词
- `boundaries`：边界/排除项
- `adjustment_strategy`：调整策略

后续所有查询生成、放宽、收窄都以该意图为不变量，避免关键词漂移。

**借鉴建议**：在 ScholarIR 的 `query_understanding` 阶段，把 `intent` 和 `boundaries` 显式输出并作为后续 query refinement 的约束。

---

## 2. 数据源解耦 + 语法隔离

PaperSeek 为每个数据源（OpenAlex、arXiv、Crossref、WoS 等）提供独立的 query generation / broaden / narrow prompt，并严格规定输出格式：

- 只输出 query string；
- 禁止输出 API 参数、字段标签、URL；
- Provider 层统一 `PaperRecord` 数据模型，上层无需关心原始 JSON。

**借鉴建议**：ScholarIR 的 `search/adapt/` 已经做了很好的适配器解耦，可进一步把「数据源查询语法规则」收敛到各 adapt 模块的 prompt/配置中，而不是散落在 pipeline 里。

---

## 3. 多信号多路召回 + RRF 融合

PaperSeek 在候选池上做多路召回，每路对应一种信号：

| Lane | 含义 | OpenAlex 实现 |
|---|---|---|
| `RELEVANCE` | 数据源原生相关性 | `sort=relevance_score:desc` |
| `IMPACT` | 影响力 | `sort=cited_by_count:desc` |
| `RECENT` | 新颖度 | `sort=publication_date:desc` |
| `LOCAL_QUALITY` | 本地质量特征 | BM25 / 词项覆盖 / venue 等 |

最终用加权 RRF 融合：

```
final_score = rrf_score + 0.35 * embedding_cosine + 0.35 * bm25_norm + 0.20 * coverage
```

**借鉴建议**：ScholarIR 的 `ranking/base.py` 目前已有 filter_score、citation_score、recency_score、venue_score、title_density_score，可继续加入 embedding cosine，并统一用 RRF 或加权融合替代简单线性加和。

---

## 4. 外部 Embedding / Reranker 的鲁棒调用策略

PaperSeek 支持通过 OpenAI-compatible API 接入外部 embedding 和 reranker，并做了多层保护：

- **多模型 fallback**：`retrieval_embedding_model` 可用逗号分隔多个模型，失败自动切换；
- **Key 复用**：未配置专用 Key 时复用 `LLM_API_KEY`；
- **Provider 自动 base URL**：支持 `openai`、`siliconflow`、`cstcloud`、`dashscope`、`zhipu`、`nvidia`、`modelscope` 等；
- **失败回退**：外部 embedding 全部失败时回退到本地预排序，不阻断主流程；
- **NVIDIA 特殊格式**：query 用 `input_type="query"`，passage 用 `"passage"`。

**借鉴建议**：ScholarIR 接入外部 embedding 时，应封装统一的 `EmbeddingClient`，支持多模型 fallback 和失败降级，并允许复用 `LLM_API_KEY`。

---

## 5. 引用网络扩展（Citation Expansion）

PaperSeek 不是简单地把引用数当特征，而是选择三类 seed 分别扩展：

1. **高相关 seed**（主要扩 references / citations）
2. **高被引 seed**（主要扩 backward references）
3. **最新发表 seed**（主要扩 forward citations）

扩展结果再与主候选池合并、去重、重新排序，并构建节点-边图供 Citation Map 使用。

**借鉴建议**：ScholarIR 目前已有单层 citation expansion，可升级为「多类型 seed + 前后向分别扩展 + 结果再排序」的策略，提升召回质量。

---

## 6. 迭代校准（Broaden / Narrow / Repair）

PaperSeek 主搜索循环最多 5 轮，根据命中情况自动调整：

- 命中为 0 或低于阈值 → `_broaden_query`
- 命中高于上限 → `_narrow_query`
- 查询被 API 拒绝（400/422/syntax 错误）→ `_repair_source_query_syntax`
- WoS 返回非标准 HTTP 512 → 降级 proximity / exact-phrase 为简单 Boolean

反馈中会附带当前返回的前 5 条标题，让 LLM 判断 on-intent / off-intent / missing facet。

**借鉴建议**：ScholarIR 当前 search 是一次性的，可在 `search` 和 `filter` 之间加入一个 controller：根据 `n_candidates`、分数分布、关键词覆盖度决定是否 broaden/narrow/repair。

---

## 7. LLM 排序的工程化降级

PaperSeek 对大批量候选做 LLM 打分时：

- 自动分 batch；
- 使用 ThreadPoolExecutor 并发；
- 并发数按 `32 → 16 → 8 → 4` 降级，避免单点超时或限流拖垮整次检索；
- 全部失败时回退到本地预排序并给 0 分，而不是直接报错。

**借鉴建议**：ScholarIR 的 `filter/base.py` 目前只把 Top-K 发给 LLM 精判一次，可引入 batch + 并发降级机制，提升在线稳定性。

---

## 8. 可审计、可复现的工作流

PaperSeek 每个阶段（intent、query、search、ranking、results）都通过 `event_handler` 发送 stage/log/quota 事件，历史记录保存在本地 SQLite，便于回溯。

**借鉴建议**：ScholarIR 已有 `trace` 字段记录各阶段信息，可进一步把 trace 持久化到 SQLite 或文件，支持查询级审计与效果复盘。

---

## 总结

最值得 ScholarIR 优先落地的三点：

1. **外部 embedding 接入 ranking 阶段**（与你当前目标一致）；
2. **搜索迭代校准**（弥补当前一次性检索的召回不足）；
3. **LLM 精判的 batch + 降级**（提升系统鲁棒性）。

其余设计（数据源解耦、RRF 融合、引用扩展策略、可审计工作流）可作为中长期改进方向参考。
