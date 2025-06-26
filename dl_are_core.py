# 파일명: dl_are_core.py
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Tuple, Dict

from gpe_decoder import GpeDecoder
from evg import ExpectationVectorGenerator

class DlAreCore:
    """
    GPE 페이로드를 받아 기대 벡터를 설정하고,
    DRIFT-LOOP를 통해 환각을 제어하며 텍스트를 생성합니다.
    """
    def __init__(self, model_name: str = "gpt2", device: str = 'cpu'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.gpe_decoder = GpeDecoder()
        self.evg = ExpectationVectorGenerator(device=self.device)
        
        self.expectation_vector: torch.Tensor | None = None
        self.ema_similarity = 0.98 # EMA 초기값
        
        print(f"✅ DL-ARE Core 초기화 완료. Generator: {model_name}")

    def initialize_with_gpe(self, gpe_payload: Dict):
        """GPE 페이로드로 제어기를 초기화합니다."""
        decoded_data = self.gpe_decoder.decode(gpe_payload)
        self.expectation_vector = self.evg.build_from_decoded_gpe(decoded_data)
        self.ema_similarity = 0.98 # 새 컨텍스트마다 EMA 리셋
        print(f"🔹 DL-ARE가 새로운 기대 벡터(E)로 초기화되었습니다. (Norm: {self.expectation_vector.norm().item():.2f})")

    @torch.no_grad()
    def generate_controlled_text(self, prompt: str, max_new_tokens: int = 50) -> Dict:
        if self.expectation_vector is None:
            raise ValueError("DL-ARE is not initialized. Call initialize_with_gpe() first.")

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = list(input_ids[0].cpu().numpy())
        reprojection_count = 0

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            
            # Drift-Loop
            similarity = F.cosine_similarity(hidden_state, self.expectation_vector.unsqueeze(0)).item()
            self.ema_similarity = 0.8 * self.ema_similarity + 0.2 * similarity # EMA 업데이트
            
            if self.ema_similarity < 0.97: # 드리프트 감지
                reprojection_count += 1
                alpha = 0.1
                hidden_state = (1 - alpha) * hidden_state + alpha * self.expectation_vector.unsqueeze(0)
            
            logits = self.model.lm_head(hidden_state)
            next_token_id = torch.argmax(logits, dim=-1)
            
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        final_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        stats = {
            "total_tokens": len(generated_ids) - input_ids.shape[1],
            "reprojection_events": reprojection_count,
            "final_ema_similarity": self.ema_similarity
        }
        return {"final_text": final_text, "generation_stats": stats}
