# 파일명: main_orchestrator.py

import asyncio
import uuid
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

# --- 서킷 브레이커 라이브러리 import ---
# 실제 환경에서는 'pip install resilience4py' 설치가 필요합니다.
# 여기서는 시뮬레이션을 위해 모의 객체를 사용합니다.
try:
    from resilience import circuit_breaker, CircuitBreaker, CircuitBreakerOpenError
except ImportError:
    # 라이브러리가 없을 경우를 위한 모의 객체
    print("WARNING: 'resilience4py' not found. Using mock CircuitBreaker.")
    class CircuitBreakerOpenError(Exception): pass
    class CircuitBreaker:
        def __init__(self, failure_threshold=3, recovery_timeout=30): pass
        def __call__(self, func):
            async def wrapper(*args, **kwargs): return await func(*args, **kwargs)
            return wrapper
    def circuit_breaker(failure_threshold=3, recovery_timeout=30):
        def decorator(func):
            async def wrapper(*args, **kwargs): return await func(*args, **kwargs)
            return wrapper
        return decorator


# --- 모든 컴포넌트 import ---
# 각 모듈이 별도 파일로 존재한다고 가정
from reasoning_engine import MockHybridReasoningEngine
from dl_are_core import DlAreCore

# --- API 데이터 모델 정의 ---
class ReasoningRequest(BaseModel):
    query: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class ResultResponse(BaseModel):
    task_id: str
    status: str
    result: Any | None = None
    error_message: Optional[str] = None

# --- 시스템 전역 객체 초기화 ---
app = FastAPI(
    title="CGA-Enterprise (v3.0.1) - Resilient Orchestrator",
    version="3.0.1"
)

# 백엔드/프론트엔드 엔진 및 태스크 저장소 초기화
# (실제 프로덕션에서는 이들을 별도의 마이크로서비스 및 Redis로 대체)
glassbox_backend: MockHybridReasoningEngine | None = None
dl_are_frontend: DlAreCore | None = None
tasks: Dict[str, Any] = {}

@app.on_event("startup")
def startup_event():
    """애플리케이션 시작 시 모든 핵심 엔진을 로드합니다."""
    global glassbox_backend, dl_are_frontend
    print("✅ CGA-Enterprise Orchestrator 초기화 시작...")
    glassbox_backend = MockHybridReasoningEngine()
    dl_are_frontend = DlAreCore(model_name="gpt2")
    print("✅ 모든 컴포넌트가 성공적으로 로드되었습니다.")

# --- 서킷 브레이커가 적용된 백엔드 호출 함수 ---
@circuit_breaker(failure_threshold=3, recovery_timeout=60)
async def call_backend_with_breaker(query: str) -> Dict[str, Any]:
    """
    서킷 브레이커를 통해 백엔드 추론 엔진을 호출합니다.
    장애 발생 시 CircuitBreakerOpenError를 발생시킵니다.
    """
    print("📞 [CircuitBreaker] 백엔드 서비스 호출 시도...")
    # asyncio.to_thread를 사용하여 동기 함수를 비동기 이벤트 루프에서 안전하게 실행
    gpe_payload = await asyncio.to_thread(glassbox_backend.reason, query)
    print("👍 [CircuitBreaker] 백엔드 서비스 호출 성공.")
    return gpe_payload

# --- 비동기 파이프라인 작업 함수 ---
async def run_full_pipeline_task(task_id: str, query: str):
    """
    백그라운드에서 전체 파이프라인을 실행하며, 서킷 브레이커를 통한 장애 복구를 포함합니다.
    """
    print(f"🔹 [Task: {task_id}] 전체 파이프라인 시작...")
    tasks[task_id] = {"status": "processing", "stage": "backend_reasoning"}
    
    gpe_payload = None
    is_fallback = False
    
    try:
        # 1. 서킷 브레이커를 통해 백엔드 호출
        gpe_payload = await call_backend_with_breaker(query)
        print(f"   [Task: {task_id}] 백엔드 추론 및 GPE 인코딩 완료.")

    except CircuitBreakerOpenError:
        print(f"🚨 [Task: {task_id}] 서킷 브레이커가 열렸습니다! 백엔드 장애 감지.")
        is_fallback = True
        # 2. 폴백(Fallback) 메커니즘: GPE 없이 단순 컨텍스트 생성
        # GPE의 부재를 알리는 특별한 페이로드 생성
        gpe_payload = {
            "payload_type": "fallback_v1.0",
            "generative_payload": {
                "raw_context": f"System is under high load. Providing a direct answer for: {query}"
            },
            "metadata": {"reason": "Backend service unavailable"}
        }
        print(f"   [Task: {task_id}] 폴백 GPE 페이로드 생성 완료.")

    except Exception as e:
        # 기타 예외 처리
        error_message = f"Unhandled error in backend: {e}"
        tasks[task_id] = {"status": "failed", "result": error_message}
        print(f"❌ [Task: {task_id}] 백엔드 작업 중 심각한 오류 발생: {e}")
        return

    # 3. 프론트엔드 제어기 실행
    tasks[task_id]["stage"] = "frontend_generation"
    try:
        # DL-ARE는 GPE 페이로드의 타입에 따라 다르게 초기화됨
        await asyncio.to_thread(dl_are_frontend.initialize_with_gpe, gpe_payload)
        
        # 제어된 텍스트 생성
        final_result = await asyncio.to_thread(
            dl_are_frontend.generate_controlled_text, prompt=query, max_new_tokens=100
        )
        if is_fallback:
            final_result["notes"] = "This response was generated in fallback mode due to backend issues."
            
        print(f"   [Task: {task_id}] 프론트엔드 제어 생성 완료.")
        tasks[task_id] = {"status": "completed", "result": final_result}
        print(f"✅ [Task: {task_id}] 모든 작업 완료. 최종 결과가 저장되었습니다.")

    except Exception as e:
        error_message = f"Error during frontend generation: {e}"
        tasks[task_id] = {"status": "failed", "result": error_message}
        print(f"❌ [Task: {task_id}] 프론트엔드 작업 실패: {e}")


# --- API 엔드포인트 ---
@app.post("/generate", response_model=TaskResponse, status_code=202)
async def request_generation(request: ReasoningRequest, background_tasks: BackgroundTasks):
    """최종 사용자 요청을 받아 전체 생성 파이프라인을 시작합니다."""
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(run_full_pipeline_task, task_id, request.query)
    return TaskResponse(
        task_id=task_id,
        status="accepted",
        message="Task accepted and is processing in the background. Check status at /results/{task_id}"
    )

@app.get("/results/{task_id}", response_model=ResultResponse)
async def get_generation_result(task_id: str):
    """task_id를 사용하여 작업 상태 및 최종 결과를 조회합니다."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return ResultResponse(
        task_id=task_id, status=task["status"], result=task.get("result"),
        error_message=task.get("result") if task.get("status") == "failed" else None
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "components_loaded": all([glassbox_backend, dl_are_frontend])}

# --- 로컬 실행을 위한 코드 ---
if __name__ == "__main__":
    import uvicorn
    print("🚀 로컬 CGA-Enterprise 서버를 시작합니다. http://127.0.0.1:8000/docs 에서 API를 테스트할 수 있습니다.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
