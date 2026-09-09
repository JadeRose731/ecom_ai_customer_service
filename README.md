# MewHelp · 电商智能客服系统(设计文档库)

以「喵喵优选」(猫用品电商)为业务背景,按章节推进的电商智能客服系统**设计文档仓库**。每章沉淀两份文档:**Spec(设计定稿)** 与 **Plan(实施计划)**,系统从纯对话起步,逐章演进到 Workflow+Agent 混合架构、RAG 混合检索、可观测性与数据飞轮、模型微调。

## 章节路线图

| 章节 | 主题 | 关键能力 |
|------|------|----------|
| [ch01](ch01/) | 纯对话 | FastAPI + LangChain 多轮对话、SSE 流式逐 token 输出、PromptTemplate 管理、结构化售后提取(`with_structured_output`)、历史裁剪与 token 预算 |
| [ch02](ch02/) | Function Calling 工具链 | `@tool` 内置工具(查物流/查订单/查 FAQ)、单轮工具调用、会话消息落库、聊天页工具轨迹徽章 |
| [ch03](ch03/) | RAG 知识库 | Markdown 结构感知切分、对话挖知识离线批处理、BGE-M3 嵌入、Milvus Lite dense 单路检索 |
| [ch04](ch04/) | 混合检索 + 重排 + 评估 | Milvus Standalone 原生 BM25 + dense 混合检索(RRF)、bge-reranker-v2-m3 重排、Query 理解、带引用/拒答/自评落池、四策略评估体系、👍/👎 满意度反馈 |
| [ch05](ch05/) | Workflow + Agent 混合架构 | LangGraph 确定性骨架(指代消解 → 意图识别 → 分流 → 置信度兜底)+ ReAct 主力 Agent |
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

## 技术栈

- **后端**:Python 3.12、FastAPI、LangChain / LangGraph、SQLAlchemy 2.0(异步)、uv
- **模型接入**:应用层统一走 OpenAI 协议直连上游,地址 / 模型名 / 密钥全在 `.env`,GPT / Claude / DeepSeek / GLM / Ollama 可换着接
- **检索**:Milvus(Standalone)+ BAAI/bge-m3 嵌入 + bge-reranker-v2-m3 重排 + BM25 混合检索
- **存储 / 基础设施**:MySQL(Docker)、Milvus + etcd + MinIO
- **可观测性**:Langfuse(trace 树、token 成本、评估趋势)
- **微调**:Transformers + hfl/roBERTa-wwm-ext
- **前端**:原生 HTML / CSS / JS(SSE 流式聊天页、后台管理页)
