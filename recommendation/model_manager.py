'''
모델 관리 모듈 - 임베딩 모델, LLM, FAISS 로딩
'''
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

import torch


class ModelManager:
    """모델 관리 클래스 - 임베딩/LLM/DB 등 모델 및 벡터스토어 관리"""
    
    def __init__(self):
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[ChatOpenAI] = None
        self.db: Optional[FAISS] = None
        self.retriever = None
    
    def initialize_embedding_model(self) -> None:
        """임베딩 모델 초기화"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용할 디바이스: {device}")
        
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BM-K/KoSimCSE-roberta-multitask",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        print(f"임베딩 모델 로딩 완료 ({device} 사용)")
    
    def initialize_llm(self) -> None:
        """LLM 모델 초기화"""
        try:
            self.llm = ChatOpenAI(model="gpt-4.1")
            print("OpenAI GPT-4 모델을 사용합니다.")
        except Exception as e:
            print(f"OpenAI 모델 로딩 실패: {e}")
            raise
    
    def load_faiss_database(self) -> None:
        """FAISS 데이터베이스 로딩"""
        if not self.embedding_model:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다.")
        
        script_dir = Path(__file__).parent
        persist_dir = script_dir.parent / "embedding" / "faiss_card_db"
        
        try:
            self.db = FAISS.load_local(
                str(persist_dir), 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
            self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
            print("FAISS DB 로딩 완료")
        except Exception as e:
            print(f"FAISS DB 로딩 실패: {e}")
            print(f"시도한 경로: {persist_dir}")
            print("임베딩을 먼저 실행하세요.")
            raise
    
    def validate_environment(self) -> None:
        """환경 변수 검증"""
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다.\n"
                "1. .env 파일에 OPENAI_API_KEY=your_api_key를 추가하세요.\n"
                "2. 또는 환경 변수로 설정하세요."
            ) 