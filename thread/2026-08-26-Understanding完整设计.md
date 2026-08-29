# ScholarIR Query Understanding 完整设计说明

> 日期：2026-08-26（修订：`recall_hints` → `slots.terms`；+ `query_skeleton` 防 query drift）  
> 状态：**设计规格（目标态 v2.1）**；文末注明与当前代码的差距  
> 相关：`2026-08-20-轻量槽位实现.md`、`2026-08-26-angle与SlotUsage-v2.1.md`  
> 代码目录：`src/scholar_ir/understanding/`

---

## 1. 目标与原则

Understanding 把自然语言问题变成可检索、可过滤、可评判的结构化输出：

```text
NL question
  → UnderstandingResult {
        intent, slots, relevance_criteria, sub_queries[]
     }
  → Retrieval（按 SubQuery.text + filters）
  → Judge（按 relevance_criteria）
```

### 设计原则

| # | 原则 |
|---|------|
| 1 | **单主 intent**：路由用；不做「伪造多 intent」来制造多样性 |
| 2 | **slots 统一承载事实 + 术语扩写**：约束型字段与 `terms[]` 同属 slots，不另开 `recall_hints` |
| 2b | **`query_skeleton`**：有序 keyword parts 拼成检索串；扩写时**整骨架填充、最多替换一个** replaceable part，禁止 lone synonym/abbrev |
| 3 | **Usage Decision 在代码里**：哪个槽进 text / filter / judge，不靠 LLM 猜 |
| 4 | **angle = 改写机制**：描述「这条 query 怎么生成」，不是「研究切面」 |
| 5 | **默认不改变信息需求**：lexical 扩写优先；decomposition 仅覆盖 gap 时触发 |
| 6 | **year/venue/negation 不占 sub-query 预算** |

---

## 2. 总流水线

```text
                    User Question
                          │
                          ▼
              ┌───────────────────────┐
              │ Round 1: Extract      │  LLM 或启发式
              │  intent               │
              │  slots                │
              │    ├─ 约束字段        │  year/venue/… 
              │    └─ terms[]         │  多术语 + 变体 + instances
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Slot Usage Decision   │  代码表（不进 LLM）
              │  → channel 映射       │
              │  → api_filters        │
              │  → judge criteria     │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Coverage Check (P1)   │  读 slots.terms[].instances
              │  Direct | Gap         │  P0：恒为 Direct
              └───────────┬───────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
     mode=lexical                mode=decomposition
     core/synonym/abbrev/entity   term → instances
            │                           │
            └─────────────┬─────────────┘
                          ▼
              ┌───────────────────────┐
              │ ExpansionTask[]       │
              │ Budget ≤ max_n (5)    │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Assemble SubQuery[]   │
              │  text + angle + mode  │
              │  + shared filters     │
              └───────────────────────┘
```

**控制权**：LLM 只抽 facts / terms；**是否扩写、怎么进 API、生成几条**由代码策略决定。

---

## 3. Intent

### 3.1 定义

**intent** = 用户检索目标的粗分类，用于：

- 是否走 metadata 通道（specific）
- 是否给 core 注入 modifier（如 survey）
- Judge 权重与解释
- **不**用于「默认拆成多个研究切面」

### 3.2 Closed set（6 类）

| intent | 含义 | 典型问法 |
|--------|------|----------|
| `survey` | 要综述 / review / overview | "survey of …", "literature review" |
| `method` | 找方法 / 技术 / 算法 | "methods for …", "approaches using …" |
| `dataset` | 找数据 / 基准 | "datasets for …", "benchmark" |
| `specific` | 找某篇 / 某工作 / 导航 | "the paper titled …", 明确作者+题 |
| `broad` | 主题探索，未指定单篇 | "papers about …" |
| `related` | related work / 相关工作 | "related work on …" |

### 3.3 设定要点

- **只选一个主 intent**（多 intent 暂缓）
- `broad` ≠ 「必须 subdomain / 多角度扩写」  
  `broad` 只表示：用户没有指定具体论文
- 未知 / 冲突 → 默认 `broad`

### 3.4 JSON

```json
"intent": "survey"
```

---

## 4. Slots（事实 + 术语）

### 4.1 定义

**slots** = 从问句抽出的全部结构化信息，分两类：

| 类型 | 字段 | 用途 |
|------|------|------|
| **约束型** | year / venue / authors / negation | filter / judge |
| **内容型** | topic / method + **`terms[]`** + **`query_skeleton`** | query_material；skeleton 定检索串形状 |

`null` = 未知，**不要过滤、不要编造**。

**不再使用顶层 `recall_hints` / `coverage`**：扩写素材与 umbrella instances 挂在对应 **term** 上。

### 4.2 约束与主题字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `topic` | `string \| null` | 主主题短语；与 `terms` 中 `role=topic` 的 `text` 应对齐 |
| `method` | `string \| null` | **仅具体技术名**；禁止泛化短语 |
| `year_from` | `int \| null` | 起始年 |
| `year_to` | `int \| null` | 截止年 |
| `venue` | `string \| null` | 会议/期刊 |
| `authors` | `string[] \| null` | 作者名列表 |
| `negation` | `string[] \| null` | 排除项 → Judge only |
| **`terms`** | `Term[] \| null` | 问句中的术语列表（可多条） |
| **`query_skeleton`** | `{parts: Part[]} \| null` | 有序 keyword parts；拼成检索串 |

### 4.3 Term 对象（`slots.terms[]`）

每个术语一条，承载**该词**的 lexical 变体与（可选）实例展开：

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `string` | 术语主写法（英文） |
| `role` | `string` | `topic` \| `method` \| `entity` \| `other` |
| `abbrev` | `string \| null` | 缩写或全称对应 |
| `synonyms` | `string[] \| null` | 同义 / 写法变体 |
| `instances` | `string[] \| null` | P1：umbrella 的成员/实例；P0 可抽但不用 |
| `coverage_gap_likely` | `bool \| null` | P1：搜上位词是否易漏「只用实例词」的文献 |

**role 约定**：

| role | 含义 |
|------|------|
| `topic` | 对应主主题；通常 1 条，与 `slots.topic` 一致 |
| `method` | 用户点名的方法约束（与 `slots.method` 对齐） |
| `entity` | 问句中出现的具名模型/数据集/算法，但未升为 method 约束 |
| `other` | 其它值得扩写的术语 |

### 4.4 为何用 `terms` 而不是顶层 `recall_hints`

| 方案 | 问题 |
|------|------|
| 顶层 `recall_hints` | 一份 synonym/abbrev 挂整句，**对不上是哪个词**；多术语时混乱 |
| **`slots.terms[]`** | 每个术语自带变体；Usage / Expansion 可按 term 路由 |

### 4.5 JSON 示例（多术语）

```json
"slots": {
  "topic": "parameter-efficient fine-tuning for RAG",
  "method": "LoRA",
  "year_from": 2020,
  "year_to": null,
  "venue": null,
  "authors": null,
  "negation": ["survey"],
  "terms": [
    {
      "text": "parameter-efficient fine-tuning",
      "role": "topic",
      "abbrev": "PEFT",
      "synonyms": ["parameter efficient fine tuning"],
      "instances": ["LoRA", "adapter tuning", "prefix tuning"],
      "coverage_gap_likely": true
    },
    {
      "text": "retrieval-augmented generation",
      "role": "entity",
      "abbrev": "RAG",
      "synonyms": ["retrieval augmented generation"],
      "instances": null,
      "coverage_gap_likely": false
    },
    {
      "text": "LoRA",
      "role": "method",
      "abbrev": null,
      "synonyms": ["low-rank adaptation"],
      "instances": null,
      "coverage_gap_likely": null
    }
  ]
}
```

### 4.6 抽取约束

- 不发明用户没说的 year / venue / method
- `method` 过宽 → 置 `null`，相关内容放进 `terms`（`role=entity` 或并入 topic term）
- `negation` 只进 Judge，**永不进**检索 text / api_filter
- `instances` 只允许 **member/instance**（PEFT→LoRA）；禁止 aspect（RAG→RAG evaluation）
- `slots.topic` / `slots.method` 与对应 `terms[].text`（role 匹配）宜一致；冲突时以约束字段为准做 Usage，terms 仍可供 lexical

### 4.7 已废弃的顶层结构

```text
❌ recall_hints: { abbrev, synonyms, entities }
❌ coverage: { is_umbrella, coverage_gap_likely, instances }

✅ slots.terms[].abbrev / synonyms / instances / coverage_gap_likely
```

### 4.8 `query_skeleton`（semantic core_text + replace spans）

**不要**把关系型问句压成 bag-of-phrases。骨架 = 保留关系的 `core_text` + 可替换 span：

| 字段 | 类型 | 说明 |
|------|------|------|
| `core_text` | string | 检索向短串；**保留** comparison / causality / condition / method→task |
| `parts[].id` | string | 如 `t0`（=`replace_spans`） |
| `parts[].text` | string | 必须是 `core_text` 的子串 |
| `parts[].required` | bool | 是否语义必需 |
| `parts[].replaceable` | bool | 是否可用 variants 替换 |
| `parts[].variants` | string[] \| null | 可替换表面形式 |

```text
# conjunctive OK
core_text: hybrid architectures reconstruction-based techniques

# comparative — 保留 better than，不要拆成 bag
✅ smaller dataset can produce better models than larger dataset in LLM pre-training
❌ LLM pre-training dataset size smaller dataset better performance

swap span "LLM" → "large language model":
  smaller dataset … in large language model pre-training
```

- Round1 LLM 应直接抽 `core_text` + non-overlapping parts；**禁止**从 terms 自动补 `auto*` span
- `terms` 与 `parts` **不必 1:1**：terms→理解/judge；parts→可安全替换的检索 span
- variants：**高置信** synonym/abbrev only（禁 X→X task/model；禁词形破坏）；宁缺毋滥
- 预算：不强制凑满 N；`max_lexical_swaps`（默认 3）+ round-robin；有余量且 `coverage_gap_likely` 才做 semantic/`conceptual`
- `angle=raw`：原始 NL；`angle=metadata`：仅 title / author；`angle=conceptual`：语义重述
- `specific`：允许 lexical；禁止 decomposition / semantic

---

## 5. Slot Usage Decision（Usage Decision）

### 5.1 定义

对每个约束字段与每个 term 的扩写字段，决定 **channel**，由**代码表**完成。

### 5.2 Channel 枚举

| channel | 含义 |
|---------|------|
| `query_material` | 参与生成 SubQuery.text |
| `api_filter` | 进检索 API 过滤（挂到每条 SubQuery.filters） |
| `post_filter` | 检索后本地过滤（API 不支持时 fallback） |
| `judge_only` | 只进 Judge / relevance_criteria |

### 5.3 Slot / Term → Channel 表（目标态）

| 来源 | channel | 生成 ExpansionTask？ | 说明 |
|------|---------|----------------------|------|
| `query_skeleton`（parts 全量 join） | query_material | `core` ×1 | 必有；parts 为 keyword，非句子 |
| `skeleton.parts[replaceable].variants` | query_material | `synonym`/`abbrev`：整骨架 + **swap 1 part** | ❌ 禁止 lone term |
| `method` / `terms[role=method]` | recall→**judge**；precision→api_filter | 通常不另开 text | 仅具体名 |
| `year_*` | api_filter | ❌ | 不占 text 预算 |
| `venue` | api_filter | ❌ | |
| `authors` | api_filter；specific→metadata | specific→`metadata` | |
| `negation` | judge_only | ❌ | |
| `terms[].synonyms` / `abbrev` | 合并进匹配 part 的 `variants` | 经 skeleton swap 生成 | 素材源，非独立 query |
| `terms[].instances` | query_material | P1：`mode=decomposition`，仍整骨架 swap | P0 忽略 |

### 5.4 recall_mode

- 默认 **`recall_mode=true`**：`method` **不进** `api_filter`
- `method` 仍可进 Judge，以及可选 1 条 `entity` 型 text
- adapt 层只读 **`SubQuery.filters`**，不从 raw `slots.method` 强行拼接

### 5.5 伪代码结构

```python
@dataclass
class SlotUsage:
    slot: str                 # "year_from" | "terms[0].synonyms" | ...
    channel: str
    fallback: str = ""

@dataclass
class RetrievalPlan:
    api_filters: dict
    slot_usages: list[SlotUsage]
    recall_mode: bool = True
```

---

## 6. Expansion Mode 与 Angle

### 6.1 两层不要混

| 概念 | 回答的问题 | 取值 |
|------|------------|------|
| **mode** | 为什么生成这条？ | `lexical` \| `decomposition` |
| **angle**（= transform） | 这条 text 怎么从种子得到？ | 见下表 |

### 6.2 Mode

| mode | 含义 | 何时 |
|------|------|------|
| `lexical` | 术语改写，**不改变**信息需求 | **P0 默认** |
| `decomposition` | 某 term 上位→`instances`，补 coverage gap | **P1**：`coverage_gap_likely` 且 instances 合法 |

```text
❌ broad → 必须 decomposition
✅ term.coverage_gap_likely + valid instances → decomposition
```

### 6.3 Angle（transform）closed set

| angle | 含义 | 典型来源 |
|-------|------|----------|
| `core` | topic 压缩 / keyword 化 | `slots.topic` / topic term |
| `synonym` | 同义 / 写法变体 | `terms[].synonyms` |
| `abbrev` | 缩写↔全称 | `terms[].abbrev` |
| `entity` | 具名实体 | `terms[role=entity\|method]`；或 P1 `instances` |
| `metadata` | title / author 导航 | `intent=specific` |

**不是 angle**：

| 旧/误用 | 正确归属 |
|---------|----------|
| survey / application / evaluation | ❌ 研究切面 |
| subdomain（aspect 义） | ❌；实例展开 → mode=decomposition |
| filter_only | ❌；SlotUsage.channel |
| intent_phrase | ❌；`modifiers=["survey"]` 挂在 core |

### 6.4 Intent modifier（非 angle）

| intent | 行为 |
|--------|------|
| `survey` | core + `modifiers=["survey"]`，angle 仍为 `core` |
| `method` / `dataset` | 无强制 aspect；可用 method/entity terms |
| `specific` | metadata + core |
| `broad` / `related` | 无默认 modifier；仅 lexical |

---

## 7. ExpansionTask 与预算

### 7.1 结构

```python
@dataclass
class ExpansionTask:
    mode: str           # lexical | decomposition
    transform: str      # core | synonym | abbrev | entity | metadata
    source: str         # e.g. "slots.query_skeleton.parts[t1]"
    text_seed: str      # 已渲染的完整 keyword 串（整骨架）
    term_index: int | None = None
    modifiers: list[str] = field(default_factory=list)
    swapped_part: str = ""   # 相对 core 替换了哪个 part id
```

### 7.2 P0 生成规则（core_text + single span swap）

```text
always:  lexical / core = query_skeleton.core_text (+ survey modifier)
for each replaceable part p whose text ⊆ core_text:
  for each variant ≠ p.text (全局预算截断):
    lexical / synonym|abbrev
    text_seed = core_text with p ← variant   # 仅换一个 span
if intent==specific:
                 + metadata(author|paper title only)
                 + raw(original NL) optional
                 lexical YES; decomposition NO
if intent==specific and NOT title/author:
                 do NOT label NL as metadata

# Anti-drift：禁止 lone synonym；Anti-flatten：关系型必须用 core_text 保 relation
# ✅ "smaller dataset … better … than larger dataset … LLM …"
# ❌ bag: "LLM pre-training dataset size smaller dataset better performance"
# ❌ lone: "LLM"
```

### 7.3 P1 追加规则

```text
for each term where coverage_gap_likely and instances validated as members:
  match term → skeleton part pid
  for each instance (预算剩余):
    ExpansionTask(
      mode=decomposition,
      transform=entity,
      text_seed=core_text with pid ← instance,
      swapped_part=pid,
    )
  # specific: never
```

### 7.4 Budget

- `max_subqueries` 默认 **5**
- 优先级：`core` > `entity` > `synonym` > `abbrev` > `metadata` > `raw`
- year / venue / negation **不占用**条数
- 去重：整串 text 小写归一


---

## 8. SubQuery 与 UnderstandingResult

### 8.1 SubQuery（目标态）

```python
@dataclass
class SubQuery:
    qid: str
    text: str
    channel: str = "keyword"               # keyword | metadata | semantic
    filters: Dict[str, Any] = field(default_factory=dict)
    angle: str = "core"                    # core|synonym|abbrev|entity|metadata
    mode: str = "lexical"                  # lexical | decomposition
    modifiers: List[str] = field(default_factory=list)
    angle_source: str = ""                 # e.g. "slots.terms[0].abbrev"
```

### 8.2 UnderstandingResult

```python
@dataclass
class UnderstandingResult:
    raw_question: str
    intent: str
    slots: Dict[str, Any]                  # 含 terms[]
    relevance_criteria: List[Dict[str, Any]]
    sub_queries: List[SubQuery]
```

### 8.3 Round1 抽取 JSON（目标）

```json
{
  "intent": "related",
  "slots": {
    "topic": "in-context learning LLM pre-training",
    "method": null,
    "year_from": null,
    "year_to": null,
    "venue": null,
    "authors": null,
    "negation": null,
    "terms": [
      {
        "text": "in-context learning",
        "role": "topic",
        "abbrev": null,
        "synonyms": null,
        "instances": null,
        "coverage_gap_likely": false,
        "required": true,
        "replaceable": false
      },
      {
        "text": "LLM",
        "role": "entity",
        "abbrev": "LLM",
        "synonyms": ["large language model", "large language models"],
        "instances": null,
        "coverage_gap_likely": false,
        "required": true,
        "replaceable": true
      },
      {
        "text": "pre-training",
        "role": "other",
        "required": true,
        "replaceable": false
      }
    ],
    "query_skeleton": {
      "parts": [
        {"id": "t0", "text": "in-context learning", "required": true, "replaceable": false, "variants": null},
        {"id": "t1", "text": "LLM", "required": true, "replaceable": true,
         "variants": ["LLM", "large language model", "large language models"]},
        {"id": "t2", "text": "pre-training", "required": true, "replaceable": false, "variants": null}
      ]
    }
  }
}
```

**不在 Round1 输出**：最终 `sub_queries`（由 Assemble 生成）。  
**不在 Round1 输出**：顶层 `recall_hints` / `coverage`。

### 8.4 端到端示例

**Q**: papers on in-context learning with LLM pre-training

| 层 | 输出 |
|----|------|
| intent | `related` |
| slots | `query_skeleton` 三 parts；LLM 带 synonyms |
| SlotUsage | skeleton→query_material；swap t1 only |
| SubQueries | ① `in-context learning LLM pre-training` ② `… large language models …` |
| anti-drift | ❌ 不会出现单独的 `LLM` / `large language models` |

**Q**: Survey of RAG methods since 2020.

| 层 | 输出 |
|----|------|
| intent | `survey` |
| slots | skeleton≈`retrieval-augmented generation`；year_from=2020；abbrev/synonyms 进 variants |
| SubQueries | ① core+survey ② synonym/abbrev（整串）；filters 均带 year_from=2020 |

---

## 9. 与 Retrieval / Judge 的接口

| 下游 | 消费什么 |
|------|----------|
| **Retrieval adapt** | `SubQuery.text` + `SubQuery.filters` |
| **Judge** | `relevance_criteria`（topic / method / negation / intent） |
| **Trace / eval** | intent, slots（含 terms + query_skeleton）, sub_queries[].angle/mode/text/filters |

---

## 10. 分期

| 阶段 | 内容 |
|------|------|
| **P0** | Slot Usage + `query_skeleton` fill/swap lexical；忽略 `instances`；Round2 仅 fallback |
| **P1** | 读 `terms[].coverage_gap_likely` + `instances` → mode=decomposition（仍整骨架 swap） |
| **P2** | 多 facet 问句级 decomposition；探索式 aspect **默认不做** |

---

## 11. 与当前代码的差距（2026-08-26，P0 已落地）

| 项 | 目标态 | 现状 |
|----|--------|------|
| Round1 | intent + slots（`terms` + `query_skeleton`） | ✅ `llm_extract` / heuristic |
| Round2 自由扩写 | 默认关 | ✅ 默认关；`use_llm_expand=True` 可开 legacy |
| angle | transform 5 类 | ✅ core/synonym/abbrev/entity/metadata |
| mode | lexical \| decomposition | ✅ 字段有；decomposition 需 `enable_decomposition=True` |
| Slot Usage + ExpansionTask | skeleton fill + single swap | ✅ `slot_usage.py` |
| assemble | task-driven（消费已渲染 text_seed） | ✅ |
| adapt 只读 filters | 是 | ✅ |

选项：`max_subqueries`、`recall_mode`、`enable_decomposition`（P1）、`use_llm_expand`（legacy）。

---

## 12. 一句话总览

> **Intent 路由；Slots 存约束 + `terms[]` + semantic `query_skeleton(core_text, parts)`；扩写在 core_text 上最多换一个 span（防 drift、保 relation）；specific 保留 lexical；`raw`≠`metadata`；angle 记改写方式；mode 区分 lexical 与 instance decomposition。**
