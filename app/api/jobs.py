# app/api/jobs.py
from fastapi import APIRouter, HTTPException

from app.core import jobs

router = APIRouter(prefix="/api/jobs")


@router.get("")
async def list_jobs() -> dict:
    return {"jobs": jobs.all_status()}


@router.post("/{name}")
async def run_job(name: str) -> dict:
    try:
        jobs.start(name)
    except jobs.JobNotFound:
        raise HTTPException(status_code=404, detail=f"未知作业:{name}")
    except jobs.JobAlreadyRunning:
        raise HTTPException(status_code=409, detail=f"作业 {name} 还在跑,拒绝重入")
    return jobs.status(name)


@router.get("/{name}")
async def job_status(name: str) -> dict:
    try:
        return jobs.status(name)
    except jobs.JobNotFound:
        raise HTTPException(status_code=404, detail=f"未知作业:{name}")


@router.post("/{name}/stop")
async def stop_job(name: str) -> dict:
    try:
        jobs.stop(name)
    except jobs.JobNotFound:
        raise HTTPException(status_code=404, detail=f"未知作业:{name}")
    return jobs.status(name)
