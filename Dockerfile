FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# 复制项目文件
COPY pyproject.toml poetry.lock* ./
COPY *.py ./
COPY test/ ./test/

# 配置 poetry 不创建虚拟环境（在容器中不需要）
RUN poetry config virtualenvs.create false

# 安装依赖
RUN poetry install --no-dev

# 运行测试和lint检查（构建时验证）
RUN poetry run pylint *.py test/*.py --fail-under=7.0
RUN poetry run pytest

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["poetry", "run", "python", "tool_agent.py"] 