# GitHub 仓库体检工具
一个基于 FastAPI + Tailwind CSS + ECharts 的 GitHub 仓库体检工具。
## 功能
- 输入公开 GitHub 仓库链接
- 获取 Star 数、Fork 数、Open Issues 数
- 展示编程语言分布饼图
- 支持 AI 综合评分和健康度分析
## 安装依赖
```powershell
python -m pip install -r requirements.txt
启动项目
python -m uvicorn app:app --reload
启动后访问：
http://127.0.0.1:8000
环境变量
请在本地创建 .env 文件：
GITHUB_TOKEN=你的GitHubToken
DEEPSEEK_API_KEY=你的DeepSeekKey
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
