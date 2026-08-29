# Thread：Understanding → API：各基线怎么转？有没有 adapt？

> 日期：2026-08-21  
> 问题：intent/slots（或等价理解结果）如何变成可调 API 的形式？别的基线有没有「一个 API 一个 adapt」？

---

## 1. 短答

| 问题 | 答案 |
|------|------|
| 基线有统一的 `adapt(source, understanding)` 吗？ | **基本没有**这种显式、对称的抽象。 |
| 那它们怎么接 API？ | **按源/按流水线写死调用**；需要时再做「抽关键词」或「把槽位填进该路参数」。 |
| 要不要我们做成「一源一 adapt」？ | **建议做**——比 SPAR 更清晰，正好接你们的 slots。 |

---

## 2. 各基线对照

### 2.1 PaSa —— 无 adapt，单通路字符串

```text
Crawler.generate_query → ["Search]...[" 字符串]
        ↓ 几乎原样
google_search_arxiv_id(q, end_date)   # Serper: q + site:arxiv.org + before:
        ↓
arxiv / 本地库补题摘
```

- **没有** intent/slots → 多 API 参数映射。  
- `end_date` 是评测泄漏控制，不是从槽位来的通用 filter 层。  
- 「理解结果」本身就是 **已可搜的 NL 检索式**。

---

### 2.2 SPAR —— 有「按源分支」，但不是干净的 adapt 接口

```text
expand_query → NL 检索式列表 + suitable_sources
        ↓
search_papers(querys, sources=[...])
        ├─ source=="arxiv"
        │     querys 原样 → Serper(NL)
        ├─ source in {openalex, pubmed}
        │     每条 query → LLM extract_keywords(query, source) → 关键词 → 该源 search(keyword)
        └─ （semantic 在 search_funcs 里有，多源调度里常未对称接入）
```

特点：

- **按源 if/elif 分支** ≈ 半套 adapt，散落在 `MultiSearchAgent.search_papers` 里。  
- 对 OA/PubMed：多一次 **源感知抽词**（文本→关键词），不是 slots→filter。  
- intent/domain/time **很少**变成 `year=` / `venue=` 参数。  
- **没有** `adapt_semantic()` / `adapt_openalex()` 这种独立函数表。

可记为：**「源路由 + 条件抽词」≠ 「结构化 adapt 层」。**

---

### 2.3 Paper Finder —— 按意图换整条子 Agent（最不像单一 adapt）

```text
Query Analyzer → AnalyzedQuery（content, authors, venues, time, criteria…）
        ↓ Planner 按 query_type
  SPECIFIC_BY_TITLE / BY_NAME / BY_AUTHOR / METADATA / BROAD(Fast|Diligent)
        ↓
  各 Agent 内部自己调 S2 / Vespa Dense / Keyword / Cohere…
  元数据：time_range、venues 在该路里直接当 filter 用
```

特点：

- 槽位 **很强**，但转换发生在 **各 pipeline 内部**，不是统一 `adapt(api)`。  
- Broad 里还有 Dense / Keyword / Snowball **多路并行**，每路自己的 query 构造。  
- 更接近：**意图路由 → 专用检索器**，而不是「同一 Understanding，多 API 适配器」。

---

### 2.4 PaperQA2 —— 工具参数级，无多学术 API adapt

```text
Agent 临场写 paper_search(query, min_year, max_year)
        ↓
本地 Tantivy 关键词检索（可带年份）
```

- 「理解」与「API 形」常常 **一步完成**（模型直接产出工具参数）。  
- 年份像 slots→filter；但目标是本地索引，不是 S2/OA。

---

## 3. 形态对比图

```text
PaSa:     Understanding≈检索式字符串 ──────────► 单一 API(Serper)
SPAR:     Understanding≈NL列表+荐源 ──┬─► arxiv: 原样
                                     └─► OA/PM: 抽关键词再搜
PF:       Understanding≈富槽位 ──► 路由到不同 Agent（内嵌各 API）
PaperQA:  Understanding≈工具调用 ──► 本地 search(query, years)
你们目标: Understanding≈intent+slots+sub_queries
                     ──► adapt_s2 / adapt_oa / adapt_serper ...
```

---

## 4. 「一个 API 一个 adapt」——基线少见，但对你们合适

基线为什么很少抽象成 adapt？

- PaSa 单源，没必要。  
- SPAR 工程演进出来的 if 分支，没抽接口。  
- Paper Finder 产品形态是 **多 pipeline**，不是「一理解结果打多库」。

你们已经定了：

- 统一 `SubQuery.text` + `slots/filters`  
- 多源学术 API  

因此显式 adapt **更贴你们的 I/O**，例如：

```text
retrieval/
  adapt/
    semantic.py   → {query, year, venue, fields, limit}
    openalex.py   → {filter, sort, per-page}
    serper_arxiv.py → {q: "text site:arxiv.org", before?}
    pubmed.py     → {term: 布尔式}   # P2
  base.py         → 选源、调 adapt、归一 PaperRef
```

每个 adapt 只做两件事：

1. **text 路径**：NL/短句 → 该源能吃的 query/keywords（可学 SPAR 抽词，或规则压缩）  
2. **slots 路径**：year/venue/authors → 该源 filter 参数（SPAR 弱、PF 强、你们应做）

---

## 5. 建议（冻结表述）

> 别的基线 **没有**标准「一 API 一 adapt」模块；SPAR 最接近「按源分支 + OA/PubMed 抽词」，Paper Finder 是「按意图换检索器」。  
> ScholarIR **主动做成 adapt 层**：Understanding 保持源无关；兼容 API 的形式 **只在 Retrieval/adapt_* 里出现**。

下一步实现：先写 `adapt_semantic` 或 `adapt_openalex` 一个，打通 `retrieve()` 非 stub。
