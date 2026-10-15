# Copilot 智能化系统

> 基于自研编排引擎的 API 工具调用 Agent 系统 —— 自然语言驱动的意图理解、工具检索、参数提取、工具调用与结果总结全流程闭环。

------

## 项目简介

Copilot 智能化系统是一个自然语言驱动的 API 工具调用编排系统。用户通过自然语言描述需求，系统自动完成：

- **意图理解**：判断单工具任务还是多工具串联任务
- **工具检索**：三级检索链路（向量召回 → 重排序 → LLM 精选）
- **参数提取与补全**：从用户输入中抽取参数，缺失时自动调用其他 API 补全
- **人工确认**：高风险操作需用户确认，支持 6 种反馈意图识别
- **工具调用**：执行 HTTP 请求，实时更新调用链图谱
- **结果总结**：将多步执行结果汇总为自然语言回答

> **设计背景**：项目需满足信创（信息技术应用创新）要求，不能依赖 LangGraph / LangChain 等国外编排框架，因此自研了一套轻量级状态机编排引擎，用 Task + MongoDB 模拟了 State / Checkpointer / Interrupt 等核心机制。

------

## 核心特性

### 1. 自研编排引擎

- 以 `Task` 文档为状态载体，MongoDB 持久化，支持断点续跑
- `TASK_STATUS_WAIT_CONFIRM` 实现 Human-in-the-loop 中断/恢复
- `_not_loop_validate` 四维循环检测，替代递归限制

### 2. 三级工具检索

| 层级     | 技术                       | 作用                   |
| :------- | :------------------------- | :--------------------- |
| 向量召回 | Milvus + text-embedding-v3 | 快速缩小候选范围       |
| 模型重排 | Qwen gte-rerank-v2         | 细粒度相关性排序       |
| LLM 精选 | Qwen / DeepSeek            | 业务语义匹配，最终决策 |

### 3. 参数提取与补全

- 三层防线：Prompt 约束 → Pydantic 类型校验 → 代码层业务校验
- 缺失参数自动生成补全查询，调用其他工具获取
- 支持重试与兜底降级

### 4. 人工确认闭环

- 高风险操作强制用户确认
- 支持 6 种意图识别：`confirm` / `abort` / `correct_params` / `correct_tool` / `clarify` / `unrelated`
- 版本号 + 审计日志，全链路可追溯

### 5. 四维循环检测

- 步数预算（max_steps=15）
- 状态签名（同工具+同结果重复 ≥3 次）
- 工具连续性（连续 3 次同工具）
- 工具链环路（A→B→C→A 周期检测）

### 6. 三层评估体系

| 层级       | 指标数 | 计算方式  | 说明                                 |
| :--------- | :----- | :-------- | :----------------------------------- |
| L1 执行层  | 8      | 程序化    | 工具选择 F1、幻觉率、参数存在性等    |
| L2 安全层  | 3      | 程序化    | 高风险确认、权限违规、审计完整性     |
| L3 质量层  | 9      | LLM-Judge | 目标达成度、事实准确性、计划合理性等 |
| HITL 意图  | 2      | 程序化    | 意图识别准确率、槽位填充完整度       |
| 跨运行统计 | 2      | 程序化    | 输出一致性、路径稳定性               |

### 7. 调用链可视化

- 实时记录 `nodes` / `edges`，前端可渲染为知识图谱
- 支持正向 API 规划与逆向补参链路展示

------

## 系统架构

text

```
┌─────────────────────────────────────────────────────────┐
│                      前端 (Web)                          │
│  输入自然语言 → 轮询任务状态 → 展示调用链图谱 & 结果       │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│                   Flask 后端 (app.py)                    │
│  /api_planning  /api_task_status  /login_user  ...      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              ApiPlanningHub (核心调度器)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ ApiSelection │  │ ParamExtract │  │ GenerateTask │  │
│  │    Hub       │  │    Hub       │  │    Hub       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  ToolUseHub  │  │ToolSummaryHub│  │  TaskManager │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────┬──────────────────────┬───────────────────────────┘
       │                      │
┌──────▼──────┐        ┌──────▼──────┐        ┌───────────┐
│   MongoDB   │        │   Milvus    │        │ DashScope │
│ (任务/工具) │        │  (工具向量) │        │ (LLM API) │
└─────────────┘        └─────────────┘        └───────────┘
```



------

## 技术栈

| 类别      | 技术                                           |
| :-------- | :--------------------------------------------- |
| 语言      | Python 3.10+                                   |
| Web 框架  | Flask                                          |
| 数据库    | MongoDB（任务/工具元数据）、Milvus（向量检索） |
| LLM       | 通义千问（Qwen）、DeepSeek                     |
| Embedding | text-embedding-v3                              |
| Reranker  | Qwen gte-rerank-v2                             |
| 参数校验  | Pydantic                                       |
| 测试      | pytest + pytest-cov                            |
| 部署      | Docker（可选）                                 |

------

## 快速开始

### 环境要求

- Python 3.10+
- MongoDB 4.0+
- Milvus 2.0+
- 有效的 DashScope API Key（或兼容 OpenAI 接口的国产大模型）

### 1. 克隆项目

bash

```
git clone <your-repo-url>
cd copilot-agent
```



### 2. 安装依赖

bash

```
pip install -r requirements.txt
```



### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

ini

```
# MongoDB
mongo_host=127.0.0.1
mongo_port=27017
mongo_db=tools

# Milvus
milvus_uri=http://localhost:19530
milvus_db_name=tool_db

# 模型配置
model_name=qwen-max
model_api_key=sk-xxxxxxxx
model_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模拟 API 密钥（用于调用外部业务 API）
sim_api_key=your-sim-api-key

# Judge 模型（评估用，需与主模型不同族）
judge_api_key=sk-yyyyyyyy
judge_base_url=https://api.deepseek.com/v1
judge_model=deepseek-chat
```



### 4. 启动服务

bash

```
python app.py
```



服务默认运行在 `http://0.0.0.0:5001`。

### 5. 上传工具定义

调用 `/upload_tool` 接口上传 OpenAPI/Swagger JSON 文件，系统会自动解析并存入 MongoDB + Milvus。

------

## API 接口

| 接口                  | 方法 | 说明                                              |
| :-------------------- | :--- | :------------------------------------------------ |
| `/api_planning`       | POST | 提交自然语言查询，返回 `task_id`；或提交人类反馈  |
| `/api_task_status`    | POST | 轮询任务状态，获取 `nodes`/`edges`/`systemOutput` |
| `/login_user`         | POST | 用户登录，返回 JWT Token                          |
| `/register_user`      | POST | 用户注册                                          |
| `/upload_tool`        | POST | 上传 OpenAPI 定义文件                             |
| `/get_all_tools`      | GET  | 获取所有已注册工具                                |
| `/delete_tool_by_ids` | POST | 按 ID 删除工具                                    |

### 请求示例

**提交查询：**

json

```
POST /api_planning
{
  "query": "先查一下苹果的产品信息，再查一下梨子的产品信息",
  "isCopilot": true,
  "isContext": false
}
```



**轮询状态：**

json

```
POST /api_task_status
{
  "task_id": "a1b2c3d4-..."
}
```



**人类反馈：**

json

```
POST /api_planning
{
  "taskId": "a1b2c3d4-...",
  "query": "立即执行"
}
```



------

## 测试

项目包含完整的测试体系，分为单元测试与集成测试。

### 运行单元测试（无需外部服务）

bash

```
cd test && python -m pytest unit/ -v
```



### 运行集成测试（需 MongoDB + Milvus）

bash

```
# 启动依赖服务
docker start mongo milvus-standalone milvus-etcd milvus-minio

# 运行集成测试
cd test && python -m pytest integration/ -v
```



### 覆盖率报告

bash

```
cd test && python -m pytest unit/ integration/ \
  --cov=../apis --cov=../param_extraction --cov=../tasks \
  --cov=../tools --cov=../prompt --cov=../entity --cov=../models \
  --cov-report=term --cov-config=../.coveragerc
```



**覆盖率现状**：整体 91%，核心模块 `api_planning_hub.py` 与 `api_selection_hub.py` 达 100%。

------

## 评估体系

项目内置三层自动化评估框架，独立于业务代码运行。

### 运行评估

bash

```
# 离线模拟（零 LLM 成本）
python -m eval.eval_runner --all

# 真实执行 + LLM-Judge
python -m eval.run_eval --with-judge

# 多轮执行，计算跨运行统计
python -m eval.run_eval --multi-run=3 --limit=10 --with-judge
```



### 质量门禁

| 等级    | 指标             | 门禁    |
| :------ | :--------------- | :------ |
| P0 红线 | 工具幻觉率       | == 0    |
| P0 红线 | 高风险确认缺失率 | == 0    |
| P0 红线 | 权限违规率       | == 0    |
| P0 红线 | 参数存在性       | ≥ 0.95  |
| P1 告警 | 工具选择 F1      | ≥ 0.85  |
| P1 告警 | 调用失败率       | ≤ 0.05  |
| P2 发版 | 目标达成度       | ≥ 4.0/5 |
| P2 发版 | 参数语义正确性   | ≥ 4.0/5 |

评估报告自动生成 Markdown 文件，存放于 `eval/reports/`。

------

## 项目结构

text

```
├── apis/                       # 核心编排模块
│   ├── api_planning_hub.py     # 总调度器
│   └── api_selection_hub.py    # 工具检索
├── param_extraction/           # 参数提取与校验
│   ├── param_extraction_hub.py
│   ├── parameter_validator.py
│   └── pydantic_bridge.py
├── tasks/                      # 任务生成与管理
│   ├── generate_task_hub.py
│   └── task_manager.py
├── tools/                      # 工具管理与调用
│   ├── tool_manager.py
│   ├── tool_use_hub.py
│   └── tool_summary_hub.py
├── prompt/                     # Prompt 模板（带版本号）
│   ├── general_prompts.py
│   └── qwen_model_prompts.py
├── entity/                     # 数据模型
│   ├── task_entity.py
│   ├── tool_entity.py
│   └── user_entity.py
├── models/                     # LLM / Embedding 封装
├── customize_milvus_wrapper/   # Milvus 封装
├── use_manager/                # 用户管理
├── utils/                      # 配置、日志、常量、tracing
├── eval/                       # 评估体系
│   ├── eval_trace.py
│   ├── eval_metrics.py
│   ├── eval_runner.py
│   └── test_cases/
├── test/                       # 测试体系
│   ├── unit/
│   └── integration/
├── app.py                      # Flask 入口
├── requirements.txt
└── README.md
```



------

## 安全与合规

- **信创适配**：全链路国产化（Qwen / DeepSeek / Milvus / DashScope）
- **注入检测**：LLM 判断 + 代码层参数校验双重防护
- **权限控制**：JWT + 基于角色的接口权限
- **审计日志**：所有人类反馈记录 `state_before` / `state_after` / `intent` / `action_taken`

------

## 联系方式

- 作者：李龙
- 邮箱：3490765269@qq.com

------

> 如果这个项目对你有帮助，欢迎点个 Star 支持一下！
