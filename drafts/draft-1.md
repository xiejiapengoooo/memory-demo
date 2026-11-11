### 记忆系统

记忆系统由实体，实体结构化信息，语义信息构成，实体和实体通过结构化信息关联，实体和语义信息也可关联。

#### 实体

是指在信息中具有明确或可区分身份的现实对象。它可以被唯一或相对稳定地识别，具有一组属性，并能够与其他实体建立关系。

比如：

- 公司：OpenAI、Google、字节跳动
- 一个人：张三、李四、马斯克
- 组织：研发团队、学校、政府机构
- 地点：北京、东京、某个办公楼
- 产品：iPhone、ChatGPT、MacBook
- 项目：某个软件开发项目
- 作品：一本书、一部电影、一首歌曲
- 技术：Python、RAG、Transformer

核心判断标准： 一个对象是否具有相对独立的身份，能否被单独指代、识别，并在不同上下文中保持身份连续性。

例如，「公司」是实体类型，而「OpenAI」是一个具体实体；「人」是实体类型，而「张三」是一个具体实体。

#### 实体结构化信息

实体结构化信息表示实体当前所处的状态。

初始化实体结构化数据如下：
```json
{
  "id": "uuid",
  "keywords": "",
  "agent_cognition": "",
  "relations": {}
}
```

字段说明：

所有字段都是 key=value，value 都为 string。

- id：必要，初始化自动生成，是 uuid。
- keywords：必要，暂无关键词，可为空字符串，多个关键词用逗号分隔。
- agent_cognition: 必要，agent 对当前实体的认知和描述。
- relations：必要，暂无关系指向，可为空字典结构。

relations 具体说明：

表示和其他实体的关系，某个关系有多个实体，用逗号隔开，比如：

- 朋友：```"relations":{"friends":"uuid-1,uuid-2"}```
- 同事：```"relations":{"friends":"uuid-3"}```

实体结构化信息存位置：

.memory，比如：实体 id=uuid -> .memory/uuid-1.json。

#### 语义信息

和实体当前状态有关的事实，偏好，目标等内容，或者是明确需要记住的信息。

比如：

1. user：我在巨核科技工作。 -> `[[实体id]]在[[实体id]]工作。`
2. user: 记住 1+1=3 -> `1+1=3`

语义信息存储位置：

.memory/memory.db

#### 记忆初始化

1. 检查 .memory 里是否有 `agent.json` 文件
2. 有 -> 跳过，没有 -> 自动生成

`agent.json`

```json
{
  "id": "agent",
  "keywords": "",
  "agent_cognition": "我是一个智能体。",
  "relations": {}
}
```

#### 记忆的提取与存储过程

在给定的信息中，识别到实体，实体关系，实体关系变更或者明确要求记忆某个信息时，需要创建和更新记忆内容。

示例如下：

用户发消息：

```
message-1-user: 你好，我在巨核科技工作，黄志鑫喜欢玩游戏，他是我的同事。
```

记忆 agent 思考：
1. 提取：（指代：「我」->「发消息的对象」「他」->「黄志鑫」）（实体：「黄志鑫」「巨核科技」）。
1. 有提取到实体，那么应该查询看看记忆中是否有相关实体的结构化信息。
2. 调用"实体结构化信息查询工具"。

执行"实体结构化信息查询工具"，返回结果：

```json
[]
```

记忆 agent 思考：

1. "实体结构化信息查询工具"执行结果交给模型。
2. 结果为空，需要新建实体。
3. 无法确定「当前发消息的对象」身份，无需对当前聊天对象建立结构化信息。
4. 生成2个实体结构化信息，生成对实体初步的认知。
5. 调用"保存结构化信息工具"

执行"保存结构化信息工具"，返回结果：

```json
{
  "id": "uuid-1",
  "keywords": "黄志鑫",
  "agent_cognition": "一个喜欢玩游戏的人",
  "relations": {},
  "name": "黄志鑫",
}
{
  "id": "uuid-2",
  "keywords": "巨核科技",
  "agent_cognition": "一家公司",
  "relations": {},
  "name": "巨核科技"
}
```

记忆 agent 思考：

1. "保存结构化信息工具"执行结果和当前聊天上下文交给模型。
2. 提取实体间关系：`「uuid-1」-> companies ->「uuid-2」` `「uuid-2」-> employees ->「uuid-1」`。
3. 生成实体更新结构化信息：`uuid-1 -> {"relations":{"companies":"uuid-2"}}` `uuid-2 -> {"relations":{"employees":"uuid-1"}}`
4. 生成语义信息：`[[uuid-1]]喜欢玩游戏，[[uuid-1]]在[[uuid-2]]工作。`
5. 调用"更新结构化信息工具"和"语义信息保存工具"

执行"更新结构化信息工具"，更新后结构化信息应该如下：

```json
{
  "id": "uuid-1",
  "keywords": "黄志鑫",
  "agent_cognition": "一个喜欢玩游戏的人",
  "relations": {
    "companies": "uuid-2"
  },
  "name": "黄志鑫",
}
{
  "id": "uuid-2",
  "keywords": "巨核科技",
  "agent_cognition": "一家公司",
  "relations": {
    "employees": "uuid-1"
  },
  "name": "巨核科技"
}
```

执行"语义信息保存工具"，执行过程如下：

 `[[uuid-1]]喜欢玩游戏，[[uuid-1]]在[[uuid-2]]工作，[[uuid-2]]的员工是[[uuid-1]]。` -> 向量化 + 关联实体 `entities=['uuid-1','uuid-2']` -> 向量数据库

主 agent 回复：

```
message-2-assistant: 你好，我是智能体。
```

用户发消息：

```
message-3-assistant: 我叫谢佳鹏，我离职了。
```

此时整个聊天记录是：

```
message-1-user: 你好，我在巨核科技工作，黄志鑫喜欢玩游戏，他是我的同事。

message-2-assistant: 你好，我是智能体。

message-3-assistant: 我叫谢佳鹏，我离职了。
```

记忆 agent 思考：

1. 提取：（指代：「我」->「发消息的对象」「他」->「黄志鑫」）（实体：「黄志鑫」「巨核科技」「谢佳鹏」）。
2. 有提取到实体，那么应该查询看看记忆中是否有相关实体的结构化信息。
2. 调用"实体结构化信息查询工具"。

执行"实体结构化信息查询工具"，返回结果：

```json
{
  "id": "uuid-1",
  "keywords": "黄志鑫",
  "agent_cognition": "一个喜欢玩游戏的人",
  "relations": {
    "companies": "uuid-2"
  },
  "name": "黄志鑫",
}
{
  "id": "uuid-2",
  "keywords": "巨核科技",
  "agent_cognition": "一家公司",
  "relations": {
    "employees": "uuid-1"
  },
  "name": "巨核科技"
}
```

记忆 agent 思考：

1. "实体结构化信息查询工具"执行结果交给模型。
2. `uuid-1` `uuid-2` 相关信息无需变更。
3. 生成1个实体结构化信息，生成对实体初步的认知。
4. 调用"保存结构化信息工具"

执行"保存结构化信息工具"，返回结果：

```json
{
  "id": "uuid-3",
  "keywords": "谢佳鹏",
  "agent_cognition": "一个没有工作的人。",
  "relations": {},
  "name": "谢佳鹏",
}
```

记忆 agent 思考：

1. "保存结构化信息工具"执行结果和当前聊天上下文交给模型。
2. 提取实体间关系：`「uuid-1」-> companies ->「uuid-2」` `「uuid-2」-> employees ->「uuid-1」`。
3. 发现3个实体的关系没有发生变化，无需更新。
4. 生成语义信息：`[[uuid-1]]喜欢玩游戏，[[uuid-1]]在[[uuid-2]]工作，[[uuid-1]]前同事是[[uuid-3]]，[[uuid-2]]的员工是[[uuid-1]]，[[uuid-2]]雇佣过[[uuid-3]]，[[uuid-3]]前同事是[[uuid-1]],[[uuid-3]]在[[uuid-2]]工作过。`
5. 调用"语义信息保存工具"

执行"语义信息保存工具"，执行过程如下：

 `[[uuid-1]]喜欢玩游戏，[[uuid-1]]在[[uuid-2]]工作，[[uuid-1]]前同事是[[uuid-3]]，[[uuid-2]]的员工是[[uuid-1]]，[[uuid-2]]雇佣过[[uuid-3]]，[[uuid-3]]前同事是[[uuid-1]],[[uuid-3]]在[[uuid-2]]工作过。` -> 向量化 + 关联实体 `entities=['uuid-1','uuid-2','uuid-3']` -> 向量数据库


 #### 记忆 agent 整体的工作流程
 
1. 从上下文中提取出实体
2. 没有 -> END，有 -> 继续
3. 查询提取到的实体
4. 结果为空 -> 生成实体结构化信息 -> 保存，结果不为空 -> 判断是否需要更新实体结构化信息 -> 更新
5. 通过分析上下文，建立或解除实体和实体两两之间的关系，并更新。
6. 根据上下文和各实体id，生成事实语义信息，并关联实体id，保存到向量数据库。

#### 整体架构

主 agent：负责完成对用户消息的回答。

记忆 agent：负责对消息内容进行记忆的提取和逻辑分析，并采用适当工具进行存储。

向量数据库：milvus

向量化：dashscope 模块 + qwen3.7-text-embedding

模型：langchain_deepseek + deepseek_flash

框架：langchain + langgraph

环境变量：系统中已经有需要的环境变量，无需 .env，模型名称直接写死在代码上

#### 目录文件说明

- .memory/: 记忆数据存储位置
- drafts/: 草稿箱
- specs/: 基于草稿箱里文件生成对应的需求文档，比如 `draft-1.md` -> `spec-1.md`
- src/memory/: 项目代码主目录
- src/memory/agents: 存放各种 agent
- src/memory/agent.py: 主 agent
- src/memory/milvus.py: 向量数据库
