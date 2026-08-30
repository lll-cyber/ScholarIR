# Filter / Ranking 当前流程

> 日期：2026-08-30  
> 对应代码：`src/scholar_ir/filter/base.py`、`src/scholar_ir/ranking/base.py`、`pipeline.py`  
> 状态：已实现（非 stub）；`pipeline.py` 注释仍可能写「骨架」，以代码为准

---

## 0. 在总流水线中的位置

```text
question
  → (1) query_understanding
  → (2a) search / retrieval     # 多源召回候选池
  → (2b) filter / judge         # 本文：粗筛 + 打分 + 可选 LLM / RefChain
  → (3) ranking                 # 本文：多特征融合重排
  → (4) organize                # 仍为透传骨架
```

`pipeline.run()` 每次开始会 `reset_rate_limiter()`（S2 限流）。

| 阶段 | 入口 | 输入 | 输出 |
|------|------|------|------|
| Filter | `filter_papers(understanding, candidates, options)` | 理解结果 + 检索候选 | `JudgeResult`：`scored` / `selected` / `paper_ids` |
| Ranking | `rank(understanding, filter_result, options)` | 理解结果 + Filter 结果 | 更新后的 `JudgeResult` |

兼容别名：`judge = filter_papers`；pipeline 中 `options["filter"]` / `options["judge"]` 均可。

---

## 1. Filter（阶段 2）

### 1.1 总流程

```text
candidates
  │
  ├─ 1. 去重（paper_id / 规范化标题）
  │
  ├─ 2. 硬规则（不过则 score=0）
  │     · 年份 year_from / year_to
  │     · 否定词整短语命中标题/摘要
  │
  ├─ 3. 关键词覆盖打分（含同义词/变体；标题命中加权）
  │
  ├─ 4. 按分排序 → Top-K 交 LLM 精判（可选，默认开）
  │
  ├─ 5. 引用扩展 RefChain（可选，默认关）
  │     · 高分种子 → S2 references / citations
  │     · 扩展篇再规则+关键词，可选 LLM
  │     · 合并回池并重排
  │
  └─ 6. intent-aware threshold + arxiv_only + max_return
        → selected / paper_ids
```

### 1.2 各步要点

**去重**  
优先 `paper_id`；无 id 则用标题去非字母数字后的规范化串。

**硬规则**

| 规则 | 开关（默认） | 行为 |
|------|--------------|------|
| 年份 | `rule_year=True` | slots 的 `year_from` / `year_to`；论文无年份则放行 |
| 否定 | `rule_negation=True` | slots.`negation` 中短语完整出现在题摘 → score=0 |

**关键词覆盖**

- 查询侧 token 来源：`raw_question`、slots（`topic` / `method` / `dataset` / `domain`）、`terms`（`text` / `abbrev` / `synonyms` / `instances`）、`query_skeleton`（`core_text` / `parts` / `variants`）、`relevance_criteria`
- 与题摘 token 求交：`coverage = |matched| / |query_tokens|`
- 标题命中额外 bonus（约 `0.15 * title_hit_ratio`），总分截到 1.0
- 无查询 token 时给中性分 0.5

**LLM 精判（默认开启）**

- 条件：`use_llm=True` 且 DeepSeek 已配置
- 对关键词分 > 0 的 Top-`llm_top_k`（默认 15）重打 0–1 分
- Prompt 含：原问、intent、`relevance_criteria`、候选题摘（摘要截断）
- 期望 JSON：`[{paper_id, score, reason}, ...]`；解析失败则保留关键词分

**引用扩展 / RefChain（默认关闭）**

- 开关：`expand_citations=False`
- 种子：分 ≥ `seed_min_score`（默认 0.3）的前 `seed_top_k`（默认 3）篇
- 每种子：`ref_limit` / `cit_limit`（默认各 5）；总扩展 ≤ `expand_max_total`（默认 30）
- 依赖 S2：`get_paper_references` / `get_paper_citations`
- 扩展篇同样规则+关键词；`use_llm_for_expanded` 时对扩展 Top-`llm_top_k_expanded` 再 LLM
- `seed_notes` 记录与种子的引用关系，供 LLM 作弱证据

**截断与门槛**

| Intent | 默认 threshold |
|--------|----------------|
| `survey` / `broad` | 0.05 |
| `method` | 0.15 |
| 其他 | 0.25 |

- 可用 `threshold` 显式覆盖
- `arxiv_only=True`（默认）：非 arXiv id 在截断前丢掉，避免占 `max_return` 名额（PaSa 友好）
- `max_return` 默认 20 → `selected` / `paper_ids`

### 1.3 常用 options

| Key | 默认 | 含义 |
|-----|------|------|
| `threshold` | 按 intent | 最低保留分 |
| `max_return` | 20 | 最终篇数 |
| `arxiv_only` | True | 只留 arXiv 形 id |
| `rule_year` / `rule_negation` | True | 硬规则 |
| `use_llm` | True | Top-K LLM 精判 |
| `llm_top_k` | 15 | LLM 篇数 |
| `llm_temperature` / `llm_max_tokens` | 0.2 / 1024 | LLM 参数 |
| `expand_citations` | **False** | 是否 RefChain |
| `seed_top_k` / `seed_min_score` | 3 / 0.3 | 扩展种子 |
| `ref_limit` / `cit_limit` | 5 / 5 | 每种子引文量 |
| `expand_max_total` | 30 | 扩展上限 |
| `use_llm_for_expanded` | True | 扩展篇是否 LLM |
| `llm_top_k_expanded` | 10 | 扩展篇 LLM 上限 |

---

## 2. Ranking（阶段 3）

### 2.1 定位

- **不做**新的细粒度语义判定；语义相关分继承 Filter。
- 在 Filter **已选中**的集合上做多特征加权重排（不复活被 Filter 丢掉的论文）。
- 形态接近 SPAR 式元数据 Reranker，而非 PaSa Selector / Paper Finder 子条件 Judge。

### 2.2 流程

```text
filter_result.selected
  → 取出对应 scored 子集
  → 对每篇算综合分
  → 按综合分降序
  → threshold（默认 0）+ arxiv_only + max_return
  → 新的 JudgeResult
```

### 2.3 特征与默认权重

\[
\begin{aligned}
\text{final} =\ & 0.50\cdot\text{filter\_score} \\
&+ 0.20\cdot\text{citation\_score} \\
&+ 0.15\cdot\text{recency\_score} \\
&+ 0.10\cdot\text{venue\_score} \\
&+ 0.05\cdot\text{title\_density}
\end{aligned}
\]

| 特征 | 计算 |
|------|------|
| `filter_score` | Filter 阶段 `ScoredPaper.score`，截到 \[0,1\] |
| `citation_score` | \(\log(1+c)/\log(1+c_{\max})\)；从 raw 读 citationCount / cited_by_count 等 |
| `recency_score` | 候选池内年份线性归一；无年份 → 0.5 |
| `venue_score` | venue/journal 子串命中顶会顶刊表 → 1.0，否则 0 |
| `title_density` | 查询 token 与标题 token 命中比例 |

查询 token 收集方式与 Filter 侧类似（`raw_question`、slots、terms、skeleton、criteria）。

### 2.4 常用 options

| Key | 默认 | 含义 |
|-----|------|------|
| `max_return` | 20 | 最终返回数 |
| `threshold` | 0.0 | 最低综合分（默认不过滤） |
| `weights` | 见上 | 可覆盖各特征权重 |
| `arxiv_only` | True | 只保留 arXiv 形 id |

---

## 3. Filter vs Ranking 分工

| | Filter | Ranking |
|--|--------|---------|
| 目标 | 谁进最终集合（抬 Precision、控灌水） | 集合内顺序 |
| 语义 | 关键词 + 可选 LLM 题摘判定 | 基本不新增语义 |
| 元数据 | 年份/否定硬规则 | 引用/年份/venue 加权 |
| 扩召回 | 可选 RefChain | 无 |
| 分档 | 连续分 + threshold，**未**显式输出高度/部分相关 | 未做分档 |

对集合 F1：Filter（含是否开扩展）影响通常大于 Ranking；Ranking 更影响列表头部与体验。

---

## 4. 与检索侧的衔接（相关配置）

Filter/Ranking 吃的是 search 候选池。当前默认：

- `DEFAULT_SOURCES = ["arxiv", "openalex"]`（默认不含 semantic，降 S2 压力）
- `semantic_max_queries`：若启用 semantic，限制前若干子查询才打 S2
- S2：`get_paper_references` / `get_paper_citations`；429 指数退避并运行时收紧限流

池子偏小（如 `per_query_topk=10`、总候选几十）时，Filter 再强也抬不动端到端 Recall——需先看 PoolRecall。

---

## 5. 已知缺口（对照赛题 / 基线）

1. **高度相关 / 部分相关**：赛题要求分档；当前只有连续分 + 截断，无显式两档输出。  
2. **细粒度相关**：更接近关键词 + 整篇 LLM 分；未做 Paper Finder 式子条件逐条判定。  
3. **RefChain 默认关**：扩召回能力有代码，评测默认路径通常不走。  
4. **Ranking 偏工程**：与调研结论一致；深层语义应继续落在 Filter/Judge，而不是把加权公式复杂化。  
5. **pipeline 注释过时**：仍可能写 stub/骨架，以本文件与源码为准。

---

## 6. 代码索引

| 模块 | 路径 |
|------|------|
| Filter 主逻辑 | `src/scholar_ir/filter/base.py` → `filter_papers` |
| Ranking 主逻辑 | `src/scholar_ir/ranking/base.py` → `rank` |
| 编排 | `src/scholar_ir/pipeline.py` → `run` |
| S2 引文 / 限流 | `src/scholar_ir/search/s2_client.py` |
| 默认源 | `src/scholar_ir/config.py` → `DEFAULT_SOURCES` |
