import json
import math
import os
from datetime import UTC, datetime
from time import time
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
CACHE_TTL_SECONDS = 300
REPO_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

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

    if "/" in cleaned_url and not cleaned_url.startswith(("http://", "https://")) and not cleaned_url.startswith("github.com"):
        cleaned_url = f"https://github.com/{cleaned_url}"
    elif not cleaned_url.startswith(("http://", "https://")):
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


def github_get_optional(path: str) -> dict[str, Any] | None:
    try:
        return github_get(path)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


def parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def days_since(value: str | None) -> int | None:
    parsed = parse_github_datetime(value)
    if not parsed:
        return None
    return max(0, (datetime.now(UTC) - parsed).days)


def get_default_ai_review() -> dict[str, Any]:
    return {
        "score": None,
        "review": "AI 评分暂不可用，已使用本地规则生成基础健康度评估。",
        "source": "unavailable",
        "label": "暂不可用",
    }


def get_score_label(score: int | None) -> str:
    if score is None:
        return "暂不可用"
    if score >= 85:
        return "优秀"
    if score >= 70:
        return "良好"
    if score >= 50:
        return "一般"
    return "需要关注"


def get_local_review(stars: int, forks: int, issues: int, languages: dict[str, Any], pushed_at: str | None) -> dict[str, Any]:
    star_score = min(30, math.log10(max(stars, 0) + 1) * 7.5)
    fork_score = min(20, math.log10(max(forks, 0) + 1) * 7)
    language_score = min(15, len([value for value in languages.values() if is_positive_number(value)]) * 3)

    inactive_days = days_since(pushed_at)
    if inactive_days is None:
        activity_score = 8
    elif inactive_days <= 30:
        activity_score = 25
    elif inactive_days <= 180:
        activity_score = 18
    elif inactive_days <= 365:
        activity_score = 10
    else:
        activity_score = 4

    issue_ratio = issues / max(stars, 1)
    issue_score = 10 if issue_ratio <= 0.02 else 7 if issue_ratio <= 0.08 else 4 if issue_ratio <= 0.2 else 1
    score = int(round(min(100, star_score + fork_score + language_score + activity_score + issue_score)))

    activity_text = "最近维护活跃" if inactive_days is not None and inactive_days <= 180 else "近期活跃度需要关注"
    language_text = "语言分布较丰富" if len(languages) >= 3 else "技术栈相对集中"
    review = "\n".join(
        [
            f"1. 仓库获得 {stars} 个 Star、{forks} 个 Fork，社区关注度处于{get_score_label(score)}水平。",
            f"2. 当前未关闭 Issue 为 {issues} 个，需结合维护节奏判断问题处理压力。",
            f"3. {activity_text}，{language_text}，建议持续观察更新频率与版本发布情况。",
        ]
    )

    return {
        "score": score,
        "review": review,
        "source": "local_fallback",
        "label": get_score_label(score),
    }


def is_positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and value > 0


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("AI 返回内容不是有效 JSON")
    return json.loads(text[start : end + 1])


def get_ai_review(stars: int, forks: int, issues: int, languages: dict[str, Any], pushed_at: str | None) -> dict[str, Any]:
    fallback_review = get_local_review(stars, forks, issues, languages, pushed_at)
    if not AI_API_KEY:
        return fallback_review

    prompt = (
        "你是一个资深开源专家。请根据以下 GitHub 仓库数据进行评估："
        f"Star数 {stars}, Fork数 {forks}, 未关闭Issue {issues}, 语言分布 {languages}。"
        "请给出一个 0-100 的综合评分，并提供 3 行简短的健康度分析。"
        "请严格以 JSON 格式返回，格式必须为：{\"score\": 85, \"review\": \"...\"}"
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
            return fallback_review

        normalized_score = max(0, min(100, int(score)))
        return {
            "score": normalized_score,
            "review": review.strip() or fallback_review["review"],
            "source": "ai",
            "label": get_score_label(normalized_score),
        }
    except Exception as exc:
        print(f"AI review failed: {type(exc).__name__}: {exc}")
        return fallback_review


def get_readme_status(owner: str, repo: str) -> dict[str, Any]:
    readme = github_get_optional(f"/repos/{owner}/{repo}/readme")
    if not readme:
        return {"exists": False, "url": None}
    return {"exists": True, "url": readme.get("html_url")}


def get_latest_release(owner: str, repo: str) -> dict[str, Any]:
    release = github_get_optional(f"/repos/{owner}/{repo}/releases/latest")
    if not release:
        return {"exists": False, "name": None, "tag_name": None, "published_at": None, "url": None}
    return {
        "exists": True,
        "name": release.get("name"),
        "tag_name": release.get("tag_name"),
        "published_at": release.get("published_at"),
        "url": release.get("html_url"),
    }


def build_repo_response(owner: str, repo: str) -> dict[str, Any]:
    cache_key = f"{owner.lower()}/{repo.lower()}"
    cached = REPO_CACHE.get(cache_key)
    if cached and time() - cached[0] < CACHE_TTL_SECONDS:
        return {**cached[1], "cache": {"hit": True, "ttl_seconds": CACHE_TTL_SECONDS}}

    repo_data = github_get(f"/repos/{owner}/{repo}")
    languages = github_get(f"/repos/{owner}/{repo}/languages")
    readme = get_readme_status(owner, repo)
    latest_release = get_latest_release(owner, repo)

    stars = int(repo_data.get("stargazers_count") or 0)
    forks = int(repo_data.get("forks_count") or 0)
    open_issues = int(repo_data.get("open_issues_count") or 0)
    pushed_at = repo_data.get("pushed_at")
    ai_review = get_ai_review(stars, forks, open_issues, languages, pushed_at)
    license_info = repo_data.get("license") or {}

    response = {
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
        "default_branch": repo_data.get("default_branch"),
        "is_archived": bool(repo_data.get("archived")),
        "is_fork": bool(repo_data.get("fork")),
        "license": {
            "name": license_info.get("name") if isinstance(license_info, dict) else None,
            "spdx_id": license_info.get("spdx_id") if isinstance(license_info, dict) else None,
        },
        "created_at": repo_data.get("created_at"),
        "updated_at": repo_data.get("updated_at"),
        "pushed_at": pushed_at,
        "days_since_push": days_since(pushed_at),
        "languages": languages,
        "readme": readme,
        "latest_release": latest_release,
        "ai": ai_review,
        "cache": {"hit": False, "ttl_seconds": CACHE_TTL_SECONDS},
    }
    REPO_CACHE[cache_key] = (time(), response)
    return response


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
