from __future__ import annotations
import json, logging
from time import perf_counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from app.jobs import enqueue_refresh, refresh_queue
from app.auth import Credentials, authenticate, current_user, register, session_secret
from app.content_collector import collect_content
from app.database import initialize_database, list_user_ids_with_profiles, read_state, write_state
from app.evaluation import compute_metrics
from app.feedback import FeedbackRequest, load_feedback, save_feedback
from app.parser import parse_export
from app.pipeline_state import persist_workspace, state_workspace
from app.rag_summarizer import generate_briefing
from app.ranker import rank_articles
from app.scheduler import create_scheduler, scheduler_enabled

ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger(__name__)

def run_pipeline(user_id: str) -> dict[str, Any]:
    started = perf_counter()
    with state_workspace(user_id, ("profile", "feedback")) as path:
        collected = collect_content(path); ranked = rank_articles(data_dir=path)
        briefing = generate_briefing(data_dir=path); metrics = compute_metrics(path)
        persist_workspace(user_id, path, ("articles", "digest", "briefing", "metrics"))
    candidates = ranked["candidate_count"]
    removed = candidates - ranked["deduplicated_count"]
    metrics.update({
        "pipeline_seconds": round(perf_counter() - started, 3),
        "near_duplicates_removed": removed,
        "near_duplicate_rate": round(removed / candidates, 4) if candidates else 0,
    })
    write_state(user_id, "metrics", metrics)
    return {"status":"completed","collected_articles":collected["article_count"],"ranked_articles":ranked["result_count"],"summary_count":briefing["summary_count"],"metrics":metrics}

def _scheduled_refresh_all() -> None:
    for user_id in list_user_ids_with_profiles():
        try: enqueue_refresh(user_id)
        except Exception: LOGGER.exception("Scheduled user refresh failed")

@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database(); scheduler = None
    if scheduler_enabled(): scheduler=create_scheduler(_scheduled_refresh_all); scheduler.start()
    yield
    if scheduler: scheduler.shutdown(wait=False)

app=FastAPI(title="FocusFeed",version="2.1.0",lifespan=lifespan)
app.add_middleware(SessionMiddleware,secret_key=session_secret(),same_site="lax",max_age=2_592_000)
app.mount("/static",StaticFiles(directory=ROOT/"app/static"),name="static")

@app.get("/",response_class=HTMLResponse)
def home()->str:return (ROOT/"app/static/index.html").read_text(encoding="utf-8")
@app.get("/health")
def health()->dict[str,Any]:return {"status":"ok","database":"postgresql","scheduler_enabled":scheduler_enabled()}
@app.post("/api/auth/register")
def create_account(body:Credentials,request:Request)->dict[str,Any]:
    user=register(body);request.session["user_id"]=user["id"];return {"user":{"email":user["email"]}}
@app.post("/api/auth/login")
def login(body:Credentials,request:Request)->dict[str,Any]:
    user=authenticate(body);request.session.clear();request.session["user_id"]=user["id"];return {"user":{"email":user["email"]}}
@app.post("/api/auth/logout")
def logout(request:Request)->dict[str,str]:request.session.clear();return {"status":"signed_out"}
@app.get("/api/auth/me")
def me(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:return {"user":{"email":user["email"]}}
@app.get("/api/profile")
def get_profile(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:return read_state(user["id"],"profile",{"interests":[]})
@app.post("/api/import")
async def import_export(file:UploadFile=File(...),user:dict[str,Any]=Depends(current_user))->dict[str,Any]:
    payload=await file.read()
    if len(payload)>100*1024*1024:raise HTTPException(413,"Archive exceeds the 100 MB limit")
    try:profile=parse_export(file.filename or "export.zip",payload)
    except (ValueError,json.JSONDecodeError) as exc:raise HTTPException(400,str(exc)) from exc
    write_state(user["id"],"profile",profile);return profile
@app.post("/api/collect")
def collect_articles(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:
    try:
        with state_workspace(user["id"],("profile",)) as path:
            result=collect_content(path);persist_workspace(user["id"],path,("articles",));return result
    except Exception as exc:raise HTTPException(502,f"Content collection failed: {exc}") from exc
@app.get("/api/articles")
def get_articles(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:return read_state(user["id"],"articles",{"article_count":0,"articles":[]})
@app.post("/api/rank")
def create_ranked_digest(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:
    try:
        with state_workspace(user["id"],("profile","articles","feedback")) as path:
            result=rank_articles(data_dir=path);persist_workspace(user["id"],path,("digest",));return result
    except Exception as exc:raise HTTPException(500,f"Ranking failed: {exc}") from exc
@app.get("/api/digest")
def get_digest(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:return read_state(user["id"],"digest",{"result_count":0,"articles":[]})
@app.post("/api/summarize")
def summarize_digest(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:
    try:
        with state_workspace(user["id"],("digest",)) as path:
            result=generate_briefing(data_dir=path);persist_workspace(user["id"],path,("briefing",));return result
    except ConnectionError as exc:raise HTTPException(503,str(exc)) from exc
    except Exception as exc:raise HTTPException(500,f"Summarization failed: {exc}") from exc
@app.get("/api/briefing")
def get_briefing(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:return read_state(user["id"],"briefing",{"summary_count":0,"items":[]})
@app.post("/api/refresh", status_code=202)
def refresh_digest(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:
    if not read_state(user["id"], "profile", {}).get("interests"):
        raise HTTPException(409, "Import an interest profile first.")
    try:
        job = enqueue_refresh(user["id"])
    except RedisError as exc:
        raise HTTPException(503, "Refresh queue unavailable.") from exc
    return {"job_id": job.id, "status": "queued"}

@app.get("/api/refresh/{job_id}")
def refresh_status(job_id: str, user: dict[str, Any] = Depends(current_user)):
    try:
        job = refresh_queue().fetch_job(job_id)
        if job is None or tuple(job.args) != (user["id"],):
            raise HTTPException(404, "Refresh job not found.")
        status = job.get_status().value
        return {"status": status, "result": job.return_value() if status == "finished" else None}
    except NoSuchJobError as exc:
        raise HTTPException(404, "Refresh job not found.") from exc
    except RedisError as exc:
        raise HTTPException(503, "Refresh queue unavailable.") from exc
@app.post("/api/feedback")
def record_feedback(body:FeedbackRequest,user:dict[str,Any]=Depends(current_user))->dict[str,Any]:
    with state_workspace(user["id"],("feedback",)) as path:
        row=save_feedback(body,path);persist_workspace(user["id"],path,("feedback",));return {"status":"saved","feedback":row}
@app.get("/api/feedback")
def get_feedback(user:dict[str,Any]=Depends(current_user))->list[dict[str,Any]]:return read_state(user["id"],"feedback",[])
@app.get("/api/metrics")
def get_metrics(user:dict[str,Any]=Depends(current_user))->dict[str,Any]:return read_state(user["id"],"metrics",{})
