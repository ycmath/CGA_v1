# 파일명: gpe_encoder.py
import json
import gzip
import base64
from typing import List, Dict, Any

class GpeEncoder:
    """
    구조적 데이터를 GPE 페이로드로 인코딩합니다.
    (간소화된 버전: 반복 규칙만 감지)
    """
    def encode(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력된 딕셔너리 데이터를 분석하고 GPE 페이로드로 변환합니다.
        """
        # 여기서는 data가 'records' 키를 가진 리스트를 포함한다고 가정
        records = data.get("records", [])
        original_size = len(json.dumps(records).encode('utf-8'))

        # 반복 규칙 감지 (간단한 예: 모든 레코드가 동일한 키를 가질 때)
        if records and all(isinstance(r, dict) for r in records):
            first_keys = set(records[0].keys())
            if all(set(r.keys()) == first_keys for r in records):
                seed = [{
                    "op": "repeat",
                    "count": len(records),
                    "instruction": {"op": "instantiate_from_template", "template_id": "common_template"}
                }]
                encoded_data = self._compress_data(records)
                encoded_size = len(json.dumps(seed).encode()) + len(encoded_data)

                return {
                    "payload_type": "gpe_v1.0",
                    "generative_payload": {
                        "seed": seed,
                        "data_context_b64_gz": encoded_data
                    },
                    "metadata": {
                        "original_size_bytes": original_size,
                        "encoded_seed_size_bytes": len(json.dumps(seed).encode()),
                        "compression_ratio": 1 - (encoded_size / original_size) if original_size > 0 else 0
                    }
                }

        # 반복 패턴을 찾지 못한 경우, 단순 압축만 적용
        encoded_data = self._compress_data(data)
        return {
            "payload_type": "gpe_v1.0_compressed_json",
            "generative_payload": {"data_b64_gz": encoded_data},
            "metadata": {"original_size_bytes": original_size}
        }

    def _compress_data(self, data: Any) -> str:
        """데이터를 JSON 직렬화 -> gzip 압축 -> base64 인코딩"""
        json_str = json.dumps(data)
        compressed = gzip.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')
