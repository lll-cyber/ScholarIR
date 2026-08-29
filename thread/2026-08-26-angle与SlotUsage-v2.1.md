# Thread：angle 重定义 + Slot Usage v2.1（收缩版 + Coverage 衔接）

> 日期：2026-08-26（修订：`core_text`+span swap；specific 保留 lexical；`raw`≠`metadata`）  
> 状态：**规划冻结（P0）**；**P1 Coverage→Decomposition 接口已定，未实现**  
> **完整规格（推荐阅读）**：`2026-08-26-Understanding完整设计.md`（含 §4.8 `query_skeleton`）  
> 前置：`2026-08-20-轻量槽位实现.md`、v2 评审收紧、Coverage Gap 讨论  
> 目标实验（P0）：验证 **Controlled Lexical Transform** 是否优于 **LLM 自由 5 条 + 研究切面 angle**

---

## 0. 共识（保留什么）

| 改动 | 结论 |
|------|------|
| `angle` = 生成/改写机制，不是 research facet | ✅ |
| Round1 material → Code 定 task → Assemble | ✅ |
| `year / venue / negation` 不占 sub-query 预算 | ✅ |
| Slot Usage Table | ✅ **P0 核心** |
| Round2 LLM 自由扩写 | ❌ fallback only |
| `broad → 强制 subdomain` | ❌ |
| subdomain 作为**默认** transform | ❌ |
| subdomain / instance expansion 作为**条件触发**能力 | ✅ **P1**，接口先定 |

---

## 1. 两种「需要 subdomain」——必须区分

### 情况 A：query 已明确（**不要拆**）

> autoregressive transformer for video generation

不需要 → video prediction / video understanding / benchmark  
（这些是 **research aspect expansion**，改变信息需求）

→ 只用 `lexical`：core + synonym + abbrev

### 情况 B：概念是 umbrella（**可能要拆**）

> papers about parameter-efficient fine-tuning

很多论文只写 LoRA / adapters / prefix tuning，**从不出现 PEFT 上位词**  
→ 存在 **coverage gap**：只搜上位词会漏文献

→ 才启动 `decomposition`：上位概念 → **代表性实例**（retrieval realization）

| 类型 | 例子 | 是否允许 |
|------|------|----------|
| **instance / member**（信息需求不变） | PEFT → LoRA, adapters | ✅ P1 |
| **research aspect**（需求被改写） | RAG → RAG evaluation；video gen → video prediction | ❌ 永不作为默认 |

**判断标准（一句话）**：

> child 是否是 parent 信息需求的**成员/实例**，且搜 child 能补上「只用上位词会漏掉」的文献？

不是「是不是看起来像 subdomain」。

---

## 2. 新概念：`Expansion Mode`（不是把 subdomain 塞回 angle）

```text
mode = lexical          # 术语改写，不改信息需求
mode = decomposition    # 上位→实例，覆盖 gap；条件触发
```

`ExpansionTask` 双维：

```python
ExpansionTask(
    mode="lexical" | "decomposition",
    transform="core" | "synonym" | "abbrev" | "entity" | "metadata",
    source="slots.topic" | "slots.terms[0].synonyms[0]" | "slots.terms[0].instances[0]",
    text_seed="...",
)
```

- **angle** 仍只描述 text 怎么生成（transform）
- **mode / task_type** 描述「为什么生成这条」——尤其 decomposition 可追溯
- SubQuery 可带 `mode` 字段做 trace：`mode="decomposition", angle="entity", angle_source="topic_decomposition"`

可选：decomposition 产出的 text 仍标 `angle="entity"`，用 `mode` + `angle_source` 区分「用户槽位实体」vs「上位概念展开的实例」。

---

## 3. 总流程（P0 + P1 接口）

```text
                    Intent + Slots（含 terms[]）
                              │
                              ▼
                   Retrieval Coverage Check
                              │
                  ┌───────────┴───────────┐
                  │                       │
           Direct sufficient         Coverage gap
           (default / P0)            (umbrella + instances)
                  │                       │
                  ▼                       ▼
           mode=lexical            mode=decomposition
           core / synonym /        representative entities
           abbrev / entity         (instance expansion)
                  │                       │
                  └───────────┬───────────┘
                              ▼
                       ExpansionTask[]
                              ▼
                       Budget (max_n=5)
                              ▼
                          SubQueries
```

与此并行：**SlotUsage** 只管 channel（query_material / api_filter / post_filter / judge），与 mode 正交。

---

## 4. Coverage Check（P1 触发条件，非 broad）

**不是** `intent == broad`。

**是**：

```text
needs_decomposition IF:
  1) topic（或 method）是 umbrella / family 概念
     AND
  2) 存在若干 child 是该概念的 retrieval realization（实例/成员）
     AND
  3) 文献中常见「只用 child 术语、不出现 parent 上位词」→ 直接 lexical 可能漏召回
```

### 允许的 children（正例）

```text
parameter-efficient fine-tuning → LoRA, adapter tuning, prefix tuning, prompt tuning
```

### 禁止的 children（反例）

```text
RAG → RAG evaluation / RAG training data     # aspect，不是实例
video generation → video prediction          # 相邻课题，不是成员
smaller dataset beats larger → scaling laws  # 系统自选研究空间
```

### Round1 可产出的 material（P1）

```json
"coverage": {
  "is_umbrella": true,
  "coverage_gap_likely": true,
  "instances": ["LoRA", "adapter tuning", "prefix tuning"]
}
```

P0：**可不抽 / 抽了也不用**（assemble 忽略 `coverage`）。  
P1：代码读 `coverage_gap_likely` + 校验 instances 后再发 `mode=decomposition` tasks。

---

## 5. P0 vs P1 分工（接上）

### P0（默认，保证不改变信息需求）

```text
topic → core + synonym + abbrev + canonical entity
mode = lexical only
Coverage Check = 恒为 Direct sufficient（或未实现）
```

### P1（条件触发）

```text
umbrella + coverage gap
→ decomposition
→ instances as entity-like sub_queries
mode = decomposition
```

目标仍是：**找同一类文献**，只是用文献里更常出现的具体词补召回。

### P2（另开，勿与 P1 混）

| 能力 | 说明 |
|------|------|
| multi-facet `query_decomposition` | 复杂问句拆成多个**子问题**（不是 umbrella→instance） |
| research aspect expansion | 明确禁止作为默认；除非产品决策要「探索式检索」 |

---

## 6. 概念分层（P0 落地）

```text
User Query
    │
    ▼
Intent + Slots
    │  topic / method / year / … / terms[]
    │       terms[].abbrev | synonyms | (instances P1)
    │
    ├──────────────────┐
    ▼                  ▼
SlotUsage[]      ExpansionTask[]   mode=lexical
 query_material     transform: core|synonym|abbrev|entity|metadata
 api_filter
 post_filter
 judge_only
    │                  │
    └────────┬─────────┘
             ▼
      Budget Selection (max_n=5)
             ▼
         SubQuery[]
```

**angle** = transform；**mode** = lexical | decomposition（P0 恒 lexical）。  
**已废弃顶层 `recall_hints` / `coverage`** → 见完整设计 §4 `slots.terms`。

---

## 7. P0 angle closed set（5 类）

| angle | 含义 | 来源 |
|-------|------|------|
| `core` | topic 压缩 | `slots.topic` |
| `synonym` | 同义/写法变体 | `slots.terms[].synonyms` |
| `abbrev` | 缩写↔全称 | `slots.terms[].abbrev` |
| `entity` | 具名实体 | `terms[role=entity|method]`；或 P1 instances |
| `metadata` | title/author 导航 | `intent=specific` |

**不在 P0**：`subdomain`, `intent_phrase`, `filter_only`  
**P1 不扩 angle**：decomposition 产出仍用 `angle=entity` + `mode=decomposition`

### intent modifier（非 angle）

`survey` → `core` + `modifiers=["survey"]`

---

## 8. Slot Usage Table（P0）

| Slot | channel | expand as task? | 说明 |
|------|---------|-----------------|------|
| **topic** | query_material | `core` ×1 | 必有 |
| **method** | judge（recall）/ api_filter（precision） | `entity` ×0–1 | 仅具体技术名 |
| **year_*** | api_filter | ❌ | 挂 filters |
| **venue** | api_filter | ❌ | |
| **authors** | api_filter + metadata | specific→metadata | |
| **negation** | judge_only | ❌ | |
| **synonyms / abbrev / entities** | query_material | 对应 transform | Round1 |

broad / related：**无默认 modifier**，仅 lexical tasks。

---

## 9. ExpansionTask 生成规则

### P0

```text
always:          lexical/core = join(query_skeleton.parts)
for each replaceable part:
  for each variant ≠ canon (budget):
    lexical/synonym|abbrev  # 整骨架 + 仅换该 part（禁止 lone term）
if survey:       modifier on core(+variants)
if specific:     metadata；不跑 LLM expand
```

**显式不做**：broad→facets；aspect expansion；Round2 自由 5 角度（仅 material 全空 fallback）；把 synonym/abbrev 单独当成一条 query

### P1（接口）

```text
if coverage.coverage_gap_likely and instances validated:
  for each instance (budget remaining):
    ExpansionTask(mode=decomposition, transform=entity, source=topic_decomposition)
```

校验：instances 必须是 parent 的 member/instance，**不是** evaluation/application/benchmark 等 aspect 标签。

---

## 10. SubQuery schema

```python
@dataclass
class SubQuery:
    qid: str
    text: str
    channel: str = "keyword"       # keyword | metadata
    filters: dict = field(default_factory=dict)
    angle: str = "core"            # core|synonym|abbrev|entity|metadata
    mode: str = "lexical"          # lexical | decomposition（P0 恒 lexical）
    modifiers: list[str] = field(default_factory=list)
    angle_source: str = ""
```

---

## 11. Round1 prompt（P0）

保留：intent, slots（含 `terms[].abbrev/synonyms`；instances 可抽不用）  
禁止：自由 sub_queries；aspect 式 subtopics；顶层 `recall_hints` / `coverage`  
LLM 不决定 year 进 filter，不决定条数。

---

## 12. 模块改造清单

| # | 模块 | P0 | P1 |
|---|------|----|----|
| 1 | `slot_usage.py` | SlotUsage + ExpansionTask(mode, transform) | Coverage→decompose tasks |
| 2 | `llm_extract.py` | 收紧 hints；可选 coverage 字段 | 强化 umbrella/instances 抽取与约束 |
| 3 | `assemble.py` | task-driven；忽略 coverage | 消费 decomposition tasks |
| 4 | `types.py` | modifiers, mode, angle_source | — |
| 5 | `llm_expand.py` | fallback only | 可选：仅补 instances |
| 6 | `base.py` | 默认不走 Round2 | 可选 Coverage Check 开关 |
| 7 | tests + eval | A/B：自由 expand vs lexical P0 | +C：开 decomposition |

---

## 13. 验证实验

| 组 | 描述 |
|----|------|
| A | 当前：Round2 自由 + 研究切面 angle |
| B | P0：lexical Controlled Transform |
| C（后） | P1：B + coverage-triggered decomposition |

指标：macro R/P/F1；spot-check 是否意图漂移（尤其 Real）。

---

## 14. 一句话总结

**subdomain 不是默认 Recall transform；它是 Coverage Gap 触发的 Decomposition 能力。**

- **P0**：`mode=lexical` — Slot Usage + core/synonym/abbrev/entity — 保证忠实  
- **P1**：仅当 umbrella + 实例能补「上位词漏召回」时 → `mode=decomposition`  
- **永不默认**：research aspect（evaluation / application / benchmark）式「多样性」
