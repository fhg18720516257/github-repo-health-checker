import json
import os
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel, Field
from starlette.requests import Request

load_dotenv()

app = FastAPI(title="GitHub 仓库体检工具 MVP")
templates = Jinja2Templates(directory="templates")

GITHUB_API_BASE = "https://api.github.com"

# AI API_KEY 配置位置：
# 推荐在 .env 中配置 DEEPSEEK_API_KEY=你的密钥。
# 如果使用 OpenAI 或其他 OpenAI 兼容模型，也可以配置 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL。
AI_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
AI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
AI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")


class RepoCheckRequest(BaseModel):
    repo_url: str = Field(..., min_length=1, description="GitHub repository URL")


def parse_github_url(repo_url: str) -> tuple[str, str]:
    """Parse GitHub repository URL and return owner/repo."""
    cleaned_url = repo_url.strip()
    if not cleaned_url:
        raise ValueError("请输入 GitHub 仓库 URL")

    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = f"https://{cleaned_url}"

    parsed = urlparse(cleaned_url)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("请输入有效的 GitHub 仓库 URL，例如：https://github.com/owner/repo")

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("URL 中缺少 owner 或 repo，例如：https://github.com/owner/repo")

    owner = parts[0].strip()
    repo = parts[1].strip().removesuffix(".git")
    if not owner or not repo:
        raise ValueError("URL 中缺少 owner 或 repo，例如：https://github.com/owner/repo")

    return owner, repo


def github_get(path: str) -> dict[str, Any]:
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "github-repo-health-mvp",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        response = requests.get(
            f"{GITHUB_API_BASE}{path}",
            headers=headers,
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="无法连接 GitHub API，请稍后重试") from exc

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="仓库不存在、不是公开仓库，或 URL 填写有误")

    if response.status_code in {403, 429}:
        rate_remaining = response.headers.get("X-RateLimit-Remaining")
        token_status = "已读取到 GITHUB_TOKEN" if github_token else "未读取到 GITHUB_TOKEN"
        if rate_remaining == "0":
            raise HTTPException(
                status_code=429,
                detail=f"GitHub API 调用次数已达上限（{token_status}）。请确认 token 设置在运行 uvicorn 的同一个 PowerShell 窗口中，并重启服务。",
            )
        raise HTTPException(status_code=403, detail=f"GitHub API 暂时拒绝访问（{token_status}），请检查 token 是否有效或稍后重试")

    if not response.ok:
        raise HTTPException(status_code=502, detail="GitHub API 请求失败，请稍后重试")

    return response.json()


def get_default_ai_review(reason: str | None = None) -> dict[str, Any]:
    review = "AI 评分暂不可用，基础仓库数据已正常返回。"
    if reason:
        review = f"{review}请查看后端终端日志排查：{reason}"

    return {
        "score": None,
        "review": review,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("AI 返回内容不是有效 JSON")
    return json.loads(text[start : end + 1])


def get_ai_review(stars: int, forks: int, issues: int, languages: dict[str, Any]) -> dict[str, Any]:
    if not AI_API_KEY:
        print("AI review skipped: DEEPSEEK_API_KEY / OPENAI_API_KEY is not configured")
        return get_default_ai_review("未读取到 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

    prompt = (
        "你是一个资深开源专家。请根据以下 GitHub 仓库数据进行评估："
        f"Star数 {stars}, Fork数 {forks}, 未关闭Issue {issues}, 语言分布 {languages}。"
        "请给出一个 0-100 的综合评分，并提供 3 行简短的健康度分析。"
        "请严格以 JSON 格式返回，格式必须为：{{\"score\": 85, \"review\": \"...\"}}"
    )

    try:
        client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)

        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "你只返回合法 JSON，不要输出 Markdown 或额外解释。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content or ""
        ai_result = extract_json_object(content)

        score = ai_result.get("score")
        review = ai_result.get("review")
        if not isinstance(score, int | float) or not isinstance(review, str):
            print(f"AI review invalid response: {ai_result}")
            return get_default_ai_review("AI 返回 JSON 字段格式不正确")

        return {
            "score": max(0, min(100, int(score))),
            "review": review.strip() or get_default_ai_review()["review"],
        }
    except Exception as exc:
        print(f"AI review failed: {type(exc).__name__}: {exc}")
        return get_default_ai_review(str(exc))


def build_repo_response(owner: str, repo: str) -> dict[str, Any]:
    repo_data = github_get(f"/repos/{owner}/{repo}")
    languages = github_get(f"/repos/{owner}/{repo}/languages")

    stars = repo_data.get("stargazers_count", 0)
    forks = repo_data.get("forks_count", 0)
    open_issues = repo_data.get("open_issues_count", 0)
    ai_review = get_ai_review(stars, forks, open_issues, languages)

    return {
        "owner": owner,
        "repo": repo,
        "name": repo_data.get("name"),
        "full_name": repo_data.get("full_name"),
        "description": repo_data.get("description"),
        "html_url": repo_data.get("html_url"),
        "stars": stars,
        "forks": forks,
        "open_issues": open_issues,
        "watchers": repo_data.get("subscribers_count", 0),
        "language": repo_data.get("language"),
        "created_at": repo_data.get("created_at"),
        "updated_at": repo_data.get("updated_at"),
        "languages": languages,
        "ai": ai_review,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/check")
def check_repo(payload: RepoCheckRequest) -> dict[str, Any]:
    try:
        owner, repo = parse_github_url(payload.repo_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_repo_response(owner, repo)


@app.get("/api/repo-health")
def repo_health(url: str = Query(..., description="GitHub repository URL")) -> dict[str, Any]:
    try:
        owner, repo = parse_github_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_repo_response(owner, repo)
