# Thread：Understanding 后续可能方向

> 日期：2026-08-22  
> 背景：DeepSeek API 已接通；Recall 有提升但 F1 仍低；与 SPAR/PaSa 互斥扩写对照讨论  
> 状态：**备忘 / 未实施**（当前仍 v0：单 intent + 2–3 条 sub_queries）

---

## 0. 当前基线（已实现）

| 项 | 现状 |
|----|------|
| intent | **单个**，closed set 6 类 |
| slots | 轻量 P0/P1；`venue/authors` 规则未抽 |
| sub_queries | LLM 成功 → 直接用 2–3 条短 keyword；失败 → `rewrite_by_intent` 模板 |
| LLM | DeepSeek v4-flash（`.env`）；启发式 fallback |
| 下游 | Retrieval 默认 arxiv；Judge stub |
| eval | `--deepseek` auto n=5 宏 Recall ≈0.25（启发式 ≈0） |

---

## 1. 多 intent（**可能，暂缓**）

**结论**：复杂真实查询里，用户可能同时带多种目标（例如「要 survey + 限定某 method + 某年」），**多 intent 在语义上是可能的**。

**为何暂缓**：

- PaSa Auto/Real 当前题面相对单一，**单主 intent + slots** 多数够用。
- 多 intent 会牵动：扩写策略冲突、Judge 权重、评测解释——收益需等 **Real/隐藏集** 或自采复杂 query 验证后再做。

**以后若做，可选形态**（择一，勿混用）：

| 方案 | 说明 |
|------|------|
| A. 主 + 次 intent | `primary_intent` + `secondary_intents[]`（如 survey + method 约束） |
| B. intent 仅路由，角度外置 | **推荐长期形态**：保持 **1 个主 intent**；Recall 多样性用 **`angle`/`tags` per sub_query**（见 §2） |
| C. 每 sub_query 一个 intent | 等价于把 intent 降级为 angle，易与现有 6 类混淆，不推荐除非重命名 |

**触发条件再开**：hidden test、PaperFinding 类多 facet 查询、或内部标注显示单 intent 误判率 > X%。

---

## 2. Recall 专用扩展（**优先候选**）

基线（SPAR/PaSa）互斥子查询：**多条 query 字符串** 为主；`reason`/`tags` 多数 **只 log，不持久化给 Judge**。

### 2.1 推荐三层（与单 intent 不矛盾）

```text
Layer 1  主 intent     → 选扩写策略（survey 偏 review，specific 偏 title…）
Layer 2  互斥 angle    → 4–5 条不同语义切面（LLM）
Layer 3  非语义 lexical → 缩写/同义词变体（RAG、BERT…），控数量
```

### 2.2 angle 不必落在 intent 枚举里

intent 是 **粗路由**；Recall 角度是 **细切面**，应允许 intent 外角度，例如：

- `application` / `evaluation` / `benchmark`
- `abbrev` / `synonym`
- `title` / `author`（navigational）
- `subtopic`（问题里隐含子 facet）

**建议 schema（未来）**：

```python
SubQuery(
  text="...",
  angle="application",   # 开放短 enum 或自由字符串
  channel="keyword",       # keyword | metadata | ...
  filters={...},           # 来自 slots；Recall 阶段 method 宜慎用过严 filter
)
```

### 2.3 是否保存 angle 给下游？

| 用途 | 是否需要持久化 angle |
|------|----------------------|
| Retrieval | 否，只用 `text` |
| trace / debug | 是 |
| Judge 分 angle 打分 | 可选 P2，非必须（基线也未做） |
| Budget / 日志 | 可选 |

---

## 3. Prompt / 参数（Recall 快赢）

- [ ] `max_subqueries` 3 → **5**（对齐 SPAR/PaSa 量级）
- [ ] Prompt：明确要求 **mutually exclusive angles** + 可选 **abbrev 行**
- [ ] `recall_mode`：Retrieval 阶段 **不传 method filter**（仅 year 等硬约束）
- [ ] 启发式 fallback：`expand_for_recall()` 在 LLM 子查询后再补 1–2 条规则变体

---

## 4. 与 intent 平行的其它 Understanding 增强（P1+）

| 方向 | 说明 |
|------|------|
| 源感知 sub_queries | 同 text 进 arxiv vs openalex 前再压关键词（学 SPAR `extract_keywords`） |
| specific / 模糊记忆 | title/author 子查询、`channel=metadata`；Real/hidden 可能更需要 |
| slots LLM 化 | venue/authors 从 NL 抽；规则补 year/negation |
| Query decomposition | 复杂问句拆成 2 个子问题（不同于 angle 扩写） |
| 本地 Qwen | vLLM 或 HF GPTQ，降 DeepSeek 成本 / 离线 eval |

---

## 5. 不在 Understanding 里做、但影响 Recall 的项

- **Serper → arXiv**（PaSa 主路，需 `GOOGLE_SERPER_KEY`）
- **双源召回** openalex + arxiv，OA 命中映射 arXiv id
- **RefChain / BFS 扩写**（SPAR Query Evolver；Understanding 之后）
- **Judge / rerank**（抬 Precision，不直接抬 Recall）

---

## 6. 建议实施顺序（Recall 阶段）

1. **P0**：Prompt + max_subqueries=5 + recall 少 filter（仅 Understanding）
2. **P0**：eval 默认 `--deepseek`，看 macro Recall / F1
3. **P1**：`SubQuery.angle` 字段 + demo 打印 trace
4. **P1**：lexical 变体（LLM 一条或规则表）
5. **P2**：Serper 或双源
6. **P2**：Judge 用 `relevance_criteria`
7. **待定**：多 intent（§1），等有更复杂评测集再定 schema

---

## 7. 参考

- 基线互斥扩写：`baselines/SPAR/instruction.py`（`template_query_fusion_*`）、`search_engine.expand_query`
- PaSa：`generate_query` → `Search]...[` 字符串列表
- 对照 thread：`2026-08-21-Understanding到API的adapt对照.md`
- IO 约定：`../技术路线/核心模块IO约定.md`
