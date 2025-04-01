# MCP_Calculator
Calculator
---

# MCP 功能模块开发文档

## 配置说明

### 示例配置

#### 方式 1
```json
{
  "mcpServers": {
    "calculator": {
      "command": "path to your python.exe",
      "args": [
        "path to the calculator.py"
      ],
      "env": {
        "PYTHONPATH": "path to your MCP_Calculator"
      }
    }
  }
}
```

#### 方式 2
```json
{
  "mcpServers": {
    "calculator": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli],numpy,scipy",
        "mcp",
        "run",
        "D:\\AI_deeplearning\\1A_large_modle\\llm_codes\\calculator\\calculater.py"
      ]
    }
  }
}
```

### 配置解析
- **`command`**: 指定Python解释器路径或运行工具（如`uv`）。
- **`args`**: 提供运行脚本所需的参数。
- **`env`**: 设置环境变量，例如`PYTHONPATH`，用于指定模块搜索路径。

---

## 装饰器功能概览

MCP 提供了多种核心装饰器，用于扩展服务器功能。以下为详细说明：

| 装饰器            | 功能描述                              |
|-------------------|---------------------------------------|
| `@app.tool()`     | 注册可调用函数作为工具                |
| `@app.prompt()`   | 注册提示模板                          |
| `@app.resource()` | 注册数据资源                          |
| `@app.list_tools()` | 定义处理工具列表请求的函数          |
| `@app.list_prompts()` | 定义处理提示模板列表请求的函数     |
| `@app.list_resources()` | 定义处理资源列表请求的函数         |

---

## 详细说明

### 1. `@app.tool()`

**功能**：将函数注册为可被客户端（如Claude）调用的工具。

**基本用法**：
```python
@app.tool(name="工具名称", description="工具描述")
def my_tool(param1: str, param2: int) -> str:
    """工具功能说明"""
    # 实现代码
    return "结果"
```

**主要特点**：
- 自动从函数签名生成参数模式。
- 支持同步和异步函数。
- 可注入`Context`对象以获取MCP功能。

---

### 2. `@app.prompt()`

**功能**：注册提示模板，用于生成结构化的对话内容。

**基本用法**：
```python
@app.prompt(name="提示名称", description="提示描述")
def my_prompt(topic: str) -> list:
    """生成关于特定主题的提示"""
    return [
        {"role": "user", "content": f"请讨论{topic}"}
    ]
```

**主要特点**：
- 支持参数化提示模板。
- 可返回单条消息或消息列表。
- 支持文本、图像和嵌入资源内容。

---

### 3. `@app.resource()`

**功能**：注册数据资源，使客户端能够访问服务器提供的数据。

**基本用法**：
```python
@app.resource("resource://my-data", name="数据名称", description="数据描述")
def get_data() -> str:
    """提供数据内容"""
    return "数据内容"
```

**主要特点**：
- 支持静态和动态生成的内容。
- 可定义参数化资源URI模板。
- 支持文本和二进制数据。
- 可设置数据的MIME类型。

---

### 4. `@app.list_tools()`

**功能**：定义处理工具列表请求的函数，当客户端查询可用工具时调用。

**基本用法**：
```python
@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """返回可用工具列表"""
    # 通常由FastMCP自动处理，除非需要自定义逻辑
    return [...]
```

**主要特点**：
- 返回已注册工具的元数据。
- 默认由FastMCP自动处理，无需手动实现。
- 仅在需要高度自定义工具列表时才需重写。

---

### 5. `@app.list_prompts()`

**功能**：定义处理提示模板列表请求的函数，当客户端查询可用提示模板时调用。

**基本用法**：
```python
@app.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    """返回可用提示模板列表"""
    # 通常由FastMCP自动处理，除非需要自定义逻辑
    return [...]
```

**主要特点**：
- 返回已注册提示模板的元数据。
- 默认由FastMCP自动处理。
- 可重写以实现动态提示列表。

---

### 6. `@app.list_resources()`

**功能**：定义处理资源列表请求的函数，当客户端查询可用资源时调用。

**基本用法**：
```python
@app.list_resources()
async def handle_list_resources() -> list[Resource]:
    """返回可用资源列表"""
    # 通常由FastMCP自动处理，除非需要自定义逻辑
    return [...]
```

**主要特点**：
- 返回已注册资源的元数据。
- 默认由FastMCP自动处理。
- 可重写以实现动态资源列表管理。

---

## 使用建议

1. **日常开发**：
   - 使用`@app.tool()`、`@app.prompt()`和`@app.resource()`即可满足大多数需求。
   
2. **高级定制**：
   - 仅在需要自定义列表行为时，才实现`@app.list_tools()`、`@app.list_prompts()`或`@app.list_resources()`。

3. **默认实现**：
   - FastMCP默认提供了这些列表处理器的实现，通常不需要自定义。

---

## 核心功能总结

这些装饰器构成了MCP的核心功能，使开发者能够创建强大的、扩展了LLM能力的应用程序。通过合理使用这些装饰器，可以快速构建高效、灵活的服务端逻辑，满足不同场景的需求。

--- 

以上文档结构清晰，适合技术团队快速理解和上手使用MCP功能模块开发。
