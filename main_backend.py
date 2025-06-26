# 파일명: main_backend.py
import asyncio
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from reasoning_engine import MockHybridReasoningEngine

# --- API 데이터 모델 ---
class ReasoningRequest(BaseModel):
    query: str

class TaskResponse(BaseModel):
    task_id: str
    status: str

class ResultResponse(BaseModel):
    task_id: str
    status: str
    result: Any | None = None

# --- 시스템 초기화 ---
app = FastAPI(
    title="CGA-Enterprise: GlassBox Backend Service",
    version="3.0"
)

# 전역 변수로 추론 엔진과 작업 결과를 저장 (실제 프로덕션에서는 Redis 등 사용)
reasoning_engine: MockHybridReasoningEngine | None = None
tasks: Dict[str, Any] = {}

@app.on_event("startup")
def startup_event():
    """애플리케이션 시작 시 추론 엔진을 로드합니다."""
    global reasoning_engine
    reasoning_engine = MockHybridReasoningEngine()
    print("✅ GlassBox Reasoning Engine이 성공적으로 로드되었습니다.")

# --- 비동기 작업 함수 ---
def run_reasoning_task(task_id: str, query: str):
    """백그라운드에서 추론 및 인코딩을 수행하는 함수"""
    print(f"🔹 [Task: {task_id}] 백그라운드 추론 작업 시작...")
    tasks[task_id] = {"status": "processing"}
    try:
        result = reasoning_engine.reason(query)
        tasks[task_id] = {"status": "completed", "result": result}
        print(f"✅ [Task: {task_id}] 작업 완료. 결과가 저장되었습니다.")
    except Exception as e:
        tasks[task_id] = {"status": "failed", "result": str(e)}
        print(f"❌ [Task: {task_id}] 작업 실패: {e}")

# --- API 엔드포인트 ---
@app.post("/request_reasoning", response_model=TaskResponse, status_code=202)
async def request_reasoning(request: ReasoningRequest, background_tasks: BackgroundTasks):
    """
    추론 요청을 접수하고, 백그라운드에서 작업을 시작합니다.
    즉시 task_id를 반환합니다.
    """
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(run_reasoning_task, task_id, request.query)
    return TaskResponse(task_id=task_id, status="accepted")

@app.get("/get_result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str):
    """
    task_id를 사용하여 작업 상태 및 결과를 조회합니다.
    """
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return ResultResponse(task_id=task_id, status=task["status"], result=task.get("result"))

@app.get("/health")
def health_check():
    return {"status": "ok", "engine_loaded": reasoning_engine is not None}

# --- 로컬 테스트를 위한 실행 코드 ---
# 이 코드는 uvicorn을 사용하여 직접 실행할 수 있습니다.
# uvicorn main_backend:app --reload
