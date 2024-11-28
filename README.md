# Agent Tools

一个基于 LangChain 的智能代理工具集，用于网页搜索、学术搜索和计算等任务。

## 功能特点

- 🔍 多源搜索支持
  - 网页搜索 
  - arXiv学术搜索
  - 本地知识库搜索
- 🧮 基础计算功能
- 🤖 智能代理调度
- 📊 结果评估系统
- 📝 详细日志记录

## 快速开始

### 环境要求

- Python 3.8+
- ChromaDB 
- LangChain
- 其他依赖见 requirements.txt

### 安装

克隆仓库:

```bash
git clone https://github.com/qqgeogor/webrag.git
```

安装依赖:

```bash
pip install -r requirements.txt
```

运行:       

``` 
python tools_agent.py

```


## components definition
### tools_agent.py
    - the main entry of the tool agent
    - gradio interface

### tools.py
    - the basic tools package, including the tools for web search, arxiv search, calculator, etc.
    - using dynamic chucking with sementic clustering to get the chunks from the web pages
    - using semantic search to find the most similar chunks from the chunks database
    - store the chunks in the ChromaDB

### agent.py
    - using agent to chain the tools
    - using prompt to control the agent 
    - using logging to record the tool usage

### prompt.py
    - stroing all the prompts

### logging_config.py
    - the logging config

### evaluation.py
    - the evaluation system



### Tools List

1. Web Search Tool
   - Search web content using search engines
   - Extract and process webpage content
   - Dynamic chunking with semantic clustering

2. arXiv Search Tool  
   - Search academic papers on arXiv
   - Extract paper metadata and abstracts
   - Filter by relevance and date

3. Calculator Tool
   - Basic arithmetic operations
   - Support for mathematical expressions
   - Error handling for invalid inputs

4. Local Knowledge Base Tool
   - ChromaDB vector storage
   - Semantic similarity search
   - Document chunking and indexing

5. Relevance Check Tool
   - Content relevance evaluation
   - Confidence scoring
   - Reason explanation

6. Answer Generator Tool
    - Generate the final answer

7. Hallucination Check Tool
    - Check the answer if it is hallucination   
    
### Agent List

1. Master Agent
   - Analyze user requests
   - Route to appropriate sub-agents
   - Handle task coordination

2. Calculator Agent
   - Process calculation requests
   - Parse mathematical expressions
   - Return formatted results

3. Web Search Agent
   - Handle web search queries
   - Filter and rank results
   - Extract relevant content

4. arXiv Agent
   - Process academic search requests
   - Filter papers by criteria
   - Format paper information

5. Tool Agent
   - Manage tool selection
   - Execute tool operations
   - Handle tool responses

6. Formater Agent
    - Format the final answer
