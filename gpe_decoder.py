# 파일명: gpe_decoder.py
import json
import gzip
import base64
from typing import Dict, Any, List

class GpeDecoder:
    """
    GPE 페이로드를 디코딩하여 원본 구조적 데이터를 재구성합니다.
    """
    def decode(self, gpe_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력된 GPE 페이로드를 분석하고 원본 데이터로 재구성합니다.
        """
        payload_type = gpe_payload.get("payload_type", "")
        generative_payload = gpe_payload.get("generative_payload", {})

        if "compressed_json" in payload_type:
            # 단순 압축된 경우
            compressed_data = generative_payload.get("data_b64_gz", "")
            return self._decompress_data(compressed_data)
        
        elif payload_type == "gpe_v1.0":
            # GPE 규칙이 포함된 경우
            seed = generative_payload.get("seed", [])
            compressed_data = generative_payload.get("data_context_b64_gz", "")
            data_context = self._decompress_data(compressed_data)
            
            # 시드 규칙 실행 (여기서는 'repeat'만 처리)
            reconstructed_data = {}
            for rule in seed:
                if rule.get("op") == "repeat":
                    # data_context의 'records'를 'count'만큼 반복 (여기서는 이미 반복된 데이터를 받았다고 가정)
                    reconstructed_data["records"] = data_context
                    reconstructed_data["conclusion"] = data_context.get("conclusion")
            return reconstructed_data
            
        return {}

    def _decompress_data(self, encoded_data: str) -> Dict[str, Any]:
        """Base64 디코딩 -> gzip 압축 해제 -> JSON 파싱"""
        if not encoded_data:
            return {}
        try:
            compressed_bytes = base64.b64decode(encoded_data)
            decompressed_json = gzip.decompress(compressed_bytes)
            return json.loads(decompressed_json.decode('utf-8'))
        except Exception:
            return {} # 디코딩 실패 시 빈 객체 반환
