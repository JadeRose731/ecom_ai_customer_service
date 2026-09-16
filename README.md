# MewHelp · 电商智能客服系统(设计文档库)

以「喵喵优选」(猫用品电商)为业务背景,按章节推进的电商智能客服系统**设计文档仓库**。每章沉淀两份文档:**Spec(设计定稿)** 与 **Plan(实施计划)**,系统从纯对话起步,逐章演进到 Workflow+Agent 混合架构、RAG 混合检索、可观测性与数据飞轮、模型微调。

## 章节路线图

| 章节 | 主题 | 关键能力 |
|------|------|----------|
| [ch01](ch01/) | 纯对话 | FastAPI + LangChain 多轮对话、SSE 流式逐 token 输出、PromptTemplate 管理、结构化售后提取(`with_structured_output`)、历史裁剪与 token 预算 |
| [ch02](ch02/) | Function Calling 工具链 | `@tool` 内置工具(查物流/查订单/查 FAQ)、单轮工具调用、会话消息落库、聊天页工具轨迹徽章 |
| [ch03](ch03/) | RAG 知识库 | Markdown 结构感知切分、对话挖知识离线批处理、BGE-M3 嵌入、Milvus Standalone dense 单路检索、`/kb` 浏览器建库 |
| [ch04](ch04/) | 混合检索 + 重排 + 评估 | Milvus Standalone 原生 BM25 + dense 混合检索(RRF)、bge-reranker-v2-m3 重排、Query 理解、带引用/拒答/自评落池、四策略评估体系、👍/👎 满意度反馈 |
| [ch05](ch05/) | Workflow + Agent 混合架构 | LangGraph 确定性骨架(指代消解 → 意图识别 → 分流 → 置信度兜底)+ ReAct 主力 Agent、七类意图 7→4 分流、SSE actions 帧(转人工/建工单) |
| [ch06](ch06/) | 意图识别与对话管理 | 指代消解 + Query 改写、多查询扩写、八类意图 + 置信度、「其他」兜底、退款售后确定性子流程、订单选择器 |
| [ch07](ch07/) | 会话上下文管理 | 滑动窗口 + 异步滚动摘要双层结构、固定拼装顺序、token 预算兜底、多会话侧栏 |
| [ch08](ch08/) | 即插即用工具系统 | 工具注册中心(内置 + MCP 统一登记)、JSON Schema 参数校验、读写权限分层、统一执行引擎、审计留痕、MCP 接入、建工单 interrupt 确认流 |
| [ch09](ch09/) | 可观测性与数据飞轮 | Langfuse trace 树、按意图 token 成本账、三入口低置信度问题池、标准化查重 + 人工审核写回知识库、自动化评估趋势 |
| [ch10](ch10/) | 模型微调 | RoBERTa-wwm-ext 全参微调 17 类主题分类器,飞轮问题池主题分布分析 |

## 目录结构

```
ch01/ … ch10/
├── Spec/        # 设计定稿:目标、范围、架构与数据流、验收标准
└── Plan/        # 实施计划:任务拆解、接口契约、逐步验收命令
ch04/            # 额外包含领域知识库文档(RAG 语料)
├── product-faq.md          # 商品与购物 FAQ
├── product-specs.md        # 商品规格手册
├── returns-policy.md       # 退货退款政策
└── after-sales-manual.md   # 售后手册
```

> 说明:本仓库为设计文档与知识库语料,代码实现按各章 Plan 在对应分支产出;敏感凭据(模型地址、API Key)一律放 `.env`(gitignore),文档中只出现占位符。

## 快速开始(当前进度:ch01 纯对话)

代码在仓库根目录,随章节演进。前置:Python 3.12 + [uv](https://docs.astral.sh/uv/)。

```bash
cp .env.example .env      # 填入 CHAT_BASE_URL / CHAT_MODEL / CHAT_API_KEY(上游认的真实模型名)
uv sync                   # 安装依赖
uv run uvicorn app.main:app --port 8000   # 或 bash scripts/dev.sh
```

### ch01 验收命令

```bash
# 1. 流式对话(SSE:逐 token data: {"delta": ...},结束 data: [DONE])
curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"session_id": "s1", "message": "你们卖猫粮吗?"}'

# 2. 两轮上下文(第二轮须复述第一轮的姓名与商品)
./scripts/demo_chat.sh

# 3. 结构化售后提取(order_id / request_type / expected_solution)
curl -s http://localhost:8000/api/extract -H 'Content-Type: application/json' \
  -d '{"text": "订单 MH20260701123 的猫爬架散架了,我要退款"}'

# 4. 标注样例评估(需服务运行中,5/5 通过)
uv run python scripts/eval_extract.py
```

```bash
uv run pytest             # 单测(20 个,不打真实模型)
```

> Windows 提示:含中文的请求体建议写入 UTF-8 文件后用 `curl --data-binary @file` 传,避免控制台 GBK 编码问题;启动前 `export PYTHONUTF8=1`(scripts/dev.sh 已内置)。

## ch02:Function Calling 工具链

在 ch01 纯对话上装「查数据」能力:5 个 `@tool` 工具(query_order / query_product / query_logistics / query_faq / create_ticket),模型经 Function Calling 自选工具,后端单轮执行回灌,最终回答仍 SSE 逐 token;会话与消息落 MySQL,聊天页显示工具轨迹徽章。

```bash
docker compose up -d      # MySQL 8(本机 3306 被占,映射到 3307)
make seed                 # 灌 faq 种子数据(幂等)
uv run uvicorn app.main:app --port 8000   # 起应用
```

验收入口:

- **聊天页** `http://localhost:8000`:问「订单 1001 的物流到哪了」→ 气泡出现「🔧 调用了 query_logistics」徽章并按结果作答;「退货政策是什么」→ query_faq 命中;「邮费是多少」→ 漏召回(预期,留 ch03 向量检索);「+ 新对话」清空会话。
- **程序化出口**:`curl -s http://localhost:8000/api/agent -H 'Content-Type: application/json' -d '{"user_id":"u1","message":"订单 1001 的物流到哪了"}'`,或 `./scripts/demo_agent.sh`。
- **标注样例评估**(7 条,核对选工具):`make eval-agent`。
- **单测**(51 个,Fake 模型不打真实上游):`uv run pytest`。

## ch03:RAG 知识库(向量语义检索)

`query_faq` 从关键词查表升级为 BGE-M3 向量检索(契约不变:命中 `{"hits":[{question,answer}]}`,未命中 `{"hits":[],"message":…}`)。文档结构感知切块(标题切 / 递归切 / 句末重叠 / 大表格按行切复制表头)与对话挖知识两条离线链路写 MySQL `knowledge_chunks`(pending),幂等 vectorize job 取 pending → 嵌入 → Milvus `knowledge` 集合(按 id upsert,崩了重跑只捡剩余 pending)→ 回填 done;在线检索直接从 Milvus 取 Top-K。浏览器建库:`/kb` 录入页(预览 dry-run / 录入按指纹幂等 / 向量化 / 检索自测),`/admin` 聚合首页。

```bash
docker compose up -d      # MySQL + Milvus 三容器(etcd / minio / standalone,2.6 起官方移除 embedded etcd)
make kb-build             # data/kb/*.md → 切块 → knowledge_chunks(pending)
make kb-vectorize         # pending → 嵌入(需 EMBED_API_KEY)→ Milvus → done(幂等可重跑)
make seed-conv            # 灌合成历史对话(喂挖知识)
make kb-mine              # 对话 → LLM 抽问答对(需聊天上游)→ 暂存表 → 去重 → pending
make eval-retrieval       # 验收:5 条换说法召回,≥4/5(需真实嵌入)
uv run uvicorn app.main:app --port 8000   # 起应用,/kb 建库、/admin 看板
```

- **单测**(93 个;Milvus 走 Docker Standalone 真连,嵌入 mock 不打真实上游):`uv run pytest`。
- **验收入口**:聊天页问「邮费是多少」→ query_faq 徽章 + 包邮答案(换说法召回);`/kb` ⑤ 检索自测看 Top-K 分数;④ 区演示中断重跑(闸条三个数就是账)。

## ch04:混合检索 + 重排 + 评估体系

`query_faq` 升级为完整 RAG 管线:Query 理解(改写+同义词扩展)→ Milvus 原生 BM25 + dense `hybrid_search`(RRF 融合,Top-50 召回)→ bge-reranker-v2-m3 精排(Top-10)→ 两道证据闸(机械:最高 rerank 分 < 0.3 判检索低置信;语义:结构化自评判证据不够)→ 首尾组装编号证据 → 拒答/落池。证据走 SSE `citations` 事件下发,聊天页 `[n]` 渲染成可点角标(浮层看 section_path + 原文),每段回答带 👍/👎 一次性反馈;两道闸都拦下的题进 `low_confidence_questions` 问题池(`retrieval_low_conf` / `self_check`)。

评估体系四策略(`vector` / `bm25` / `hybrid` / `hybrid_rerank`)对照,评估的就是线上的——全部走同一个 `search_knowledge(strategy=...)`:**300 题五桶评估集**(A 政策 / B 型号易混族 / C 口语 / D 库外应有拒答 / **E 跨文档**——一问要两三块不同小节的知识,`expect_sections_all` 分组给分),三段报告(检索 Recall@5/MRR + 证据覆盖度确定性机械匹配;生成段答案覆盖度/忠实度/拒答率带超时与单点隔离,上游挂了 generation=null 照样落盘)。评估集由 `make eval-check` 守门(五桶各 60 / 小节与要点逐字校准 / D 桶规则)。判出的编造个案进 `faith_cases` 台账(一题一行跨轮累计 + 人工处置 + 证据快照,`/rag-eval` 页面可处置);`hallucination_round` 保证判出率只算本轮,台账旧账不摊到新轮;`make judge-check` 用台账快照原样重放,人工处置当标准答案回归裁判一致率(裁判 `temperature=0` 去抖,口径七类豁免 + 六个 few-shot 判例)。

```bash
make milvus-up             # Milvus 三容器(healthz 就绪门)
make kb-build && make kb-vectorize   # 44 块语料(faq/policy/manual/spec 四源,含易混型号族)
make eval-check            # 评估集守门(改题后必跑,不需 MySQL/Milvus)
make eval-rag              # 四策略三段报告 → data/ch04/reports/rag_eval.{txt,json}
                           #   无 key 阶段可 EVAL_STRATEGIES=bm25 只跑确定性 BM25 行;
                           #   EVAL_SKIP_GENERATION=1 跳过生成段(dev 省 ~45 分钟空烧)
make judge-check           # 裁判一致性回归(需真实上游 + 台账已有已处置个案)
uv run uvicorn app.main:app --port 8000   # /rag-eval 看报告/台账 + 重跑;/ 聊天页引用角标
```

> 无 key 阶段注意:跑全量单测会把 Milvus 集合清掉(测试夹具每例 drop/重建),之后跑真集合评估前先 `make dev-vectors` 重灌(真实文本 + 确定性占位向量,BM25 真实;不动 MySQL status,真 key 的 `kb-vectorize` 原位覆盖)。

- **单测**(139 个):`uv run pytest`(Milvus Standalone 真连;LLM/嵌入/重排 mock)。
- **验收入口**:`/rag-eval` 报告页(KPI/五桶分组柱状图/读图句/汇总表 best 行底色/编造个案台账)与终端 `make eval-rag` 同一份产物;`/admin` 第五张卡看最佳 MRR。

## ch05:Workflow + Agent 混合架构(LangGraph)

LangGraph 确定性骨架 + ReAct 主力 Agent:指代消解(透传,ch06 正式版)→ 七类意图分类(单标签结构化输出)→ 7→4 分流(商品咨询/退款退货→知识路,物流/订单/售后→Agent,投诉→安抚+两可选项,闲聊→固定话术)。知识路强制检索(复用 ch04 hybrid_rerank + 两道证据闸,证据强注入 Agent system 弱则兜底拒答+落池);业务路 ReAct 环(步数封顶 6,token 只记账不截停),`create_ticket` 在环内被拦截为「建议动作」不写库;会话状态经 AsyncSqliteSaver 按 thread_id 跨轮持久。转人工/建工单走 SSE `actions` 帧 → 前端渲染两按钮,建工单弹表单(类别必选+描述必填)由用户确认后才 `POST /api/actions/create-ticket` 写库。

```bash
uv run uvicorn app.main:app --port 8000   # lifespan 起图(checkpointer=data/ch05_checkpoints.sqlite)
make eval-ch05            # 五验收端到端(需全服务起 + 真实上游)
uv run pytest             # 单测(163 个,图节点/runtime/API 全 Fake 不打真上游)
```

- **验收入口**:聊天页问「我要投诉」→ 安抚话术 + 「转人工」「建工单」两按钮(转人工为纯前端模拟,建工单走表单弹窗,成功回显工单号);问「订单1001的物流到哪了」→ Agent 自调 query_logistics 徽章;问「你好呀」→ 固定话术零工具;复杂问(尾号+发货)→ ReAct 多步后收敛。
- **接口红线已钉死**(scripts/smoke_langgraph.py):`astream(stream_mode=["messages","updates"])` 产出 `(mode, chunk)`,delta 按 `metadata.langgraph_node` 过滤,`updates` 拿 citations/actions。

## ch06:意图识别与对话管理(coref 改写 + 退款子流程)

骨架升级:指代消解正式版(`resolve_reference`:coref + 口语归一,trace 记 `coref` 是否改写)→ 八类意图 + 置信度分类(`classify(query, history) -> dict`,解析失败/越界落「其他」)→ 5 出口分流(退款退货进确定性子流程 `fetch_order → retrieve_policy(多查询扩写+去重合并)→ 主力 Agent 判可退性`;「其他」/闲聊走 `script_reply` 固定话术)。退款缺订单号时节点内 `interrupt` 弹订单选择器,前端点选 → `/api/actions/resume` 续跑;判可退时 `submit_refund` 拦截为「提交退款工单」按钮 → 表单(原因五选一)→ `POST /api/actions/create-refund` 落库。

```bash
docker compose up -d      # MySQL + Milvus 三容器
docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch06-ticket-type.sql
                          # tickets.ticket_type 枚举加「退款」(本章环境已应用,SHOW COLUMNS 见四值中文枚举)
make smoke-interrupt      # interrupt/resume 中断 surface 红线冒烟(纯 interrupt 节点,不需要上游)
make eval-ch06            # 四验收端到端(需全服务起 + 真实 key)—— 待key
uv run pytest             # 单测(195 个,图节点/runtime/API 全 Fake 不打真上游)
```

- **三个 prompt eval**(纯 Prompt 评估集,需真实 key)—— 待key:`PYTHONUTF8=1 uv run python -m scripts.eval_intent`(八类判对率 + JSON 越界应为 0 + 退款后多轮漂移应回物流)、`python -m scripts.eval_coref`(透传类必须原样不改、补全类须含上文实体)、`python -m scripts.eval_expand`(核心场景扩写应 3/3 且侧重点不同)。
- **验收入口**:聊天页不带订单号问「我要退款」→ 聊天流弹订单选择器卡片 → 点一张续跑(检索政策 → Agent 判能否退)→ 能退出「提交退款工单」按钮 → 表单提交显示退款单号(前端 interrupt→resume→退款表单全链路);多轮「订单1001到哪了 → 那我想把它退了 → 算了它现在到哪了」意图随上下文漂移,应用日志 grep `ch05 turn` 看 `route=refund_flow` 与 trace `coref` 改写明细——真实验收待 key,命令就位(`make eval-ch06` + 三个 prompt eval)。
- **接口红线已钉死**(scripts/smoke_interrupt.py):中断时 `ainvoke` 正常返回带 `__interrupt__` 键、`Command(resume=…)` 从被中断节点头重跑、流式中断以 updates chunk 浮出;langgraph 升级后先重跑此冒烟。

## 技术栈

- **后端**:Python 3.12、FastAPI、LangChain / LangGraph、SQLAlchemy 2.0(异步)、uv
- **模型接入**:应用层统一走 OpenAI 协议直连上游,地址 / 模型名 / 密钥全在 `.env`,GPT / Claude / DeepSeek / GLM / Ollama 可换着接
- **检索**:Milvus(Standalone)+ BAAI/bge-m3 嵌入 + bge-reranker-v2-m3 重排 + BM25 混合检索
- **存储 / 基础设施**:MySQL(Docker)、Milvus + etcd + MinIO
- **可观测性**:Langfuse(trace 树、token 成本、评估趋势)
- **微调**:Transformers + hfl/roBERTa-wwm-ext
- **前端**:原生 HTML / CSS / JS(SSE 流式聊天页、后台管理页)
