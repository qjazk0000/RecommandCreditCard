import os
import json
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


@dataclass  # 임베딩 관련 설정값을 저장하는 데이터 클래스
class EmbeddingConfig:
    """임베딩 설정 클래스"""
    card_folder: str = str((Path(__file__).parent / "cards").resolve())
    faiss_persist_dir: str = str((Path(__file__).parent / "faiss_card_db").resolve())
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask"
    batch_size: int = 32
    progress_interval: int = 10


class CardEmbeddingProcessor:  # 카드 JSON → 벡터 임베딩 및 FAISS DB 저장 담당 클래스
    """카드 정보 임베딩 처리 클래스"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.embedding_model = None
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_environment(self):
        """환경 변수 로드"""
        load_dotenv()
        self.logger.info("환경 변수 로드 완료")
    
    def initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        self.logger.info(f"HuggingFace 임베딩 모델 로딩 중: {self.config.embedding_model_name}")
        
        # GPU 사용 가능 여부 확인
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"사용할 디바이스: {device}")
        if device == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model_name,
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": self.config.batch_size
                }
            )
            self.logger.info(f"HuggingFace 임베딩 모델 로딩 완료 ({device} 사용)")
        except Exception as e:
            self.logger.error(f"임베딩 모델 로딩 실패: {e}")
            raise
    
    def collect_json_files(self) -> List[str]:
        """JSON 파일 수집 및 통계 생성"""
        self.logger.info(f"카드 JSON 파일 검색 중: {self.config.card_folder}")
        
        json_files = []
        issuer_stats = {}
        
        card_path = Path(self.config.card_folder)
        if not card_path.exists():
            raise FileNotFoundError(f"카드 폴더를 찾을 수 없습니다: {self.config.card_folder}")
        
        for json_file in card_path.rglob("*.json"):
            json_files.append(str(json_file))
            issuer = json_file.parent.name
            issuer_stats[issuer] = issuer_stats.get(issuer, 0) + 1
        
        self._print_collection_stats(json_files, issuer_stats)
        return json_files
    
    def _print_collection_stats(self, json_files: List[str], issuer_stats: Dict[str, int]):
        """수집 통계 출력"""
        self.logger.info(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
        self.logger.info("카드사별 파일 개수:")
        for issuer, count in sorted(issuer_stats.items()):
            self.logger.info(f"  - {issuer}: {count}개")
    
    def create_document_from_card(self, card_data: dict) -> List[Document]:
        """카드 데이터를 여러 Document로 변환 (혜택과 유의사항을 개별 chunk로 분리)"""
        documents = []
        
        # 기본 카드 정보
        card_name = card_data.get("card_name", "")
        issuer = card_data.get("issuer", "")
        card_id = card_data.get("card_id", "")
        
        # 기본 메타데이터
        base_metadata = {
            "inactive": card_data.get("inactive", ""),
            "issuer": issuer,
            "card_name": card_name,
            "annual_fee": str(card_data.get("annual_fee", "")),
            "brands": ", ".join(card_data.get("brands", [])),
            "card_id": card_id,
            "card_url": card_data.get("card_url", ""),
            "content_type": "card_info"
        }
        
        # 카드 기본 정보 Document
        card_info_content = f"카드명: {card_name}\n카드사: {issuer}\n연회비: {card_data.get('annual_fee', '')}\n브랜드: {', '.join(card_data.get('brands', []))}"
        documents.append(Document(page_content=card_info_content, metadata=base_metadata))
        
        # 혜택들을 개별 Document로 생성
        for i, benefit in enumerate(card_data.get("benefits", [])):
            benefit_type = benefit.get("type", "")
            summary = benefit.get("summary", "")
            details = "\n".join(benefit.get("details", []))
            
            benefit_content = f"[혜택 유형] {benefit_type}\n[요약] {summary}\n[상세 내용]\n{details}"
            
            benefit_metadata = base_metadata.copy()
            benefit_metadata.update({
                "content_type": "benefit",
                "benefit_type": benefit_type,
                "benefit_index": i,
                "benefit_summary": summary
            })
            
            documents.append(Document(page_content=benefit_content, metadata=benefit_metadata))
        
        # 유의사항들을 개별 Document로 생성
        for i, caution in enumerate(card_data.get("cautions", [])):
            if caution.strip():  # 빈 문자열이 아닌 경우만
                caution_content = f"[유의사항] {caution}"
                
                caution_metadata = base_metadata.copy()
                caution_metadata.update({
                    "content_type": "caution",
                    "caution_index": i,
                    "caution_text": caution
                })
                
                documents.append(Document(page_content=caution_content, metadata=caution_metadata))
        
        return documents
    
    def process_json_files(self, json_files: List[str]) -> Tuple[List[Document], Dict[str, int]]:
        """JSON 파일들을 Document로 변환"""
        self.logger.info("=== JSON 파일 임베딩 처리 시작 ===")
        
        documents = []
        stats = {"success": 0, "error": 0, "total": len(json_files), "total_documents": 0, "total_benefits": 0, "total_cautions": 0}
        
        for i, file_path in enumerate(json_files, 1):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    card_data = json.load(f)
                    card_documents = self.create_document_from_card(card_data)
                    documents.extend(card_documents)
                    
                    # 통계 업데이트
                    stats["success"] += 1
                    stats["total_documents"] += len(card_documents)
                    
                    # 혜택과 유의사항 개수 계산
                    benefits_count = len(card_data.get("benefits", []))
                    cautions_count = len([c for c in card_data.get("cautions", []) if c.strip()])
                    stats["total_benefits"] += benefits_count
                    stats["total_cautions"] += cautions_count
                
                if i % self.config.progress_interval == 0 or i == len(json_files):
                    progress = (i / len(json_files)) * 100
                    self.logger.info(
                        f"진행률: {i}/{len(json_files)} ({progress:.1f}%) - "
                        f"성공: {stats['success']}, 실패: {stats['error']}, "
                        f"총 문서: {stats['total_documents']}개"
                    )
                    
            except Exception as e:
                stats["error"] += 1
                self.logger.error(f"파일 처리 실패 {file_path}: {e}")
        
        self.logger.info("=== 임베딩 처리 완료 ===")
        self.logger.info(f"성공: {stats['success']}개, 실패: {stats['error']}개")
        self.logger.info(f"총 생성된 문서: {stats['total_documents']}개")
        self.logger.info(f"총 혜택 개수: {stats['total_benefits']}개")
        self.logger.info(f"총 유의사항 개수: {stats['total_cautions']}개")
        return documents, stats
    
    def save_to_faiss_db(self, documents: List[Document]) -> bool:
        """FAISS DB에 저장"""
        if not documents:
            self.logger.warning("저장할 문서가 없습니다.")
            return False
        
        if not self.embedding_model:
            self.logger.error("임베딩 모델이 초기화되지 않았습니다.")
            return False
        
        self.logger.info("=== FAISS DB 저장 시작 ===")
        self.logger.info(f"저장할 문서 수: {len(documents)}개")
        self.logger.info(f"저장 경로: {self.config.faiss_persist_dir}")
        
        try:
            # 기존 DB가 있다면 삭제 (충돌 방지)
            import shutil
            if os.path.exists(self.config.faiss_persist_dir):
                self.logger.info("기존 FAISS DB 삭제 중...")
                shutil.rmtree(self.config.faiss_persist_dir)
            
            self.logger.info("새로운 FAISS DB 생성 중...")
            db = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )
            
            self.logger.info("FAISS DB 저장 중...")
            db.save_local(self.config.faiss_persist_dir)
            
            # 저장 검증
            try:
                # FAISS는 저장된 파일의 존재 여부로 검증
                index_file = os.path.join(self.config.faiss_persist_dir, "index.faiss")
                if os.path.exists(index_file):
                    self.logger.info("=== FAISS DB 저장 완료 ===")
                    self.logger.info(f"입력 문서 수: {len(documents)}개")
                    self.logger.info("✅ 모든 문서가 성공적으로 저장되었습니다!")
                else:
                    self.logger.warning("⚠️  경고: FAISS 인덱스 파일이 생성되지 않았습니다.")
            except Exception as count_error:
                self.logger.warning(f"저장 검증 실패: {count_error}")
                self.logger.info("FAISS DB 저장은 완료되었습니다.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"FAISS DB 저장 실패: {e}")
            return False
    
    def run(self) -> bool:
        """전체 임베딩 프로세스 실행"""
        try:
            self.logger.info("=== 카드 정보 임베딩 프로세스 시작 ===")
            
            self.load_environment()
            self.initialize_embedding_model()
            
            json_files = self.collect_json_files()
            if not json_files:
                self.logger.error("❌ 처리할 JSON 파일을 찾을 수 없습니다.")
                return False
            
            documents, stats = self.process_json_files(json_files)
            success = self.save_to_faiss_db(documents)
            
            self.logger.info("=== 전체 프로세스 완료 ===")
            return success
            
        except Exception as e:
            self.logger.error(f"프로세스 실행 중 오류 발생: {e}")
            return False


def main():  # 카드 임베딩 전체 파이프라인 실행 함수
    """메인 실행 함수"""
    config = EmbeddingConfig()
    processor = CardEmbeddingProcessor(config)
    
    success = processor.run()
    if success:
        print("🎉 임베딩 프로세스가 성공적으로 완료되었습니다!")
    else:
        print("❌ 임베딩 프로세스 실행 중 오류가 발생했습니다.")


if __name__ == "__main__":
    main()
