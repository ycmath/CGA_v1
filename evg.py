# 파일명: evg.py
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List

class ExpectationVectorGenerator:
    """
    구조적 컨텍스트를 단일한 '기대 벡터(E)'로 변환합니다.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=self.device)
        print(f"✅ EVG 초기화 완료. Encoder: {model_name}")

    @torch.no_grad()
    def build_from_decoded_gpe(self, decoded_data: Dict) -> torch.Tensor:
        """
        디코딩된 GPE 데이터로부터 기대 벡터를 생성합니다.
        """
        conclusion = decoded_data.get("conclusion", "")
        records = decoded_data.get("records", [])

        # 텍스트 정보 수집
        all_texts = []
        weights = []

        if conclusion:
            all_texts.append(conclusion)
            weights.append(0.6) # 결론에 높은 가중치

        if records:
            # 모든 레코드의 텍스트 값을 추출하여 하나의 문자열로 결합
            record_texts = " ".join([str(v) for r in records for v in r.values()])
            all_texts.append(record_texts)
            weights.append(0.4)
            
        if not all_texts:
            return torch.zeros(self.encoder.get_sentence_embedding_dimension(), device=self.device)

        # 임베딩 및 가중 평균
        embeddings = self.encoder.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)
        
        if len(weights) == embeddings.shape[0]:
            weighted_avg_vector = torch.nn.functional.normalize(
                torch.sum(embeddings * torch.tensor(weights, device=self.device).view(-1, 1), dim=0),
                p=2, dim=0
            )
        else: # 가중치 적용이 어려운 경우 단순 평균
            weighted_avg_vector = torch.nn.functional.normalize(embeddings.mean(dim=0), p=2, dim=0)
            
        return weighted_avg_vector
