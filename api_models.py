# 파일명: api_models.py

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class ReasoningRequest(BaseModel):
    """/request_reasoning 엔드포인트의 요청 바디"""
    query: str
    user_id: Optional[str] = None
    # (향후 확장) 특정 프레임이나 설정을 함께 전달 가능
    config_overrides: Optional[Dict] = None

class TaskResponse(BaseModel):
    """/request_reasoning 엔드포인트의 즉각적인 응답"""
    task_id: str
    status: str = "processing"
    message: str = "Reasoning task has been accepted and is processing in the background."

class GpePayload(BaseModel):
    """GPE 인코딩된 페이로드의 구조"""
    payload_type: str = "gpe_v1.0"
    generative_payload: Dict[str, Any]
    metadata: Dict[str, Any]

class ResultResponse(BaseModel):
    """/get_result 엔드포인트의 최종 응답"""
    task_id: str
    status: str # "processing", "completed", "failed"
    result: Optional[Any] # 작업 완료 시, GPE 페이로드 또는 최종 텍스트
    error_message: Optional[str] = None
