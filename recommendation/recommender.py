'''
카드 추천 시스템 모듈 - RAG 체인 구성 및 실행
'''
from typing import Dict, Optional
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .model_manager import ModelManager
from .query_filter import QueryFilter
from .formatter import DocumentFormatter
from .prompt_builder import PromptBuilder


class CardRecommendationSystem:
    """카드 추천 시스템 메인 클래스 - RAG 체인 구성 및 실행"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.rag_chain = None
    
    def initialize(self) -> None:
        """시스템 초기화"""
        print("모델 초기화 중...")
        
        # 환경 변수 검증
        self.model_manager.validate_environment()
        
        # 모델 초기화
        self.model_manager.initialize_embedding_model()
        self.model_manager.initialize_llm()
        self.model_manager.load_faiss_database()
        
        # RAG 체인 구성
        self._build_rag_chain()
        
        print("모델 초기화 완료")
    
    def _build_rag_chain(self) -> None:
        """RAG 체인 구성"""
        if not self.model_manager.llm:
            raise ValueError("LLM이 초기화되지 않았습니다.")
        
        self.rag_chain = (
            RunnableLambda(self._retrieve_and_filter_docs)
            | RunnableLambda(self._format_docs_for_prompt)
            | PromptBuilder.create_recommendation_prompt()
            | self.model_manager.llm
            | StrOutputParser()
        )
    
    def _retrieve_and_filter_docs(self, query: str) -> Dict:
        """문서 검색 및 필터링"""
        if not self.model_manager.retriever:
            raise ValueError("Retriever가 초기화되지 않았습니다.")
        
        docs = self.model_manager.retriever.invoke(query)
        print(f"벡터 DB에서 {len(docs)}개 카드 검색 완료")
        
        conditions = QueryFilter.extract_filter_conditions(query)
        filtered_docs = QueryFilter.apply_filters(docs, conditions)
        
        if len(filtered_docs) != len(docs):
            print(f"필터링 후 {len(filtered_docs)}개 카드로 축소")
        
        return {"query": query, "docs": filtered_docs}
    
    def _format_docs_for_prompt(self, docs_data: Dict) -> Dict:
        """프롬프트용 문서 포맷팅"""
        query = docs_data["query"]
        docs_list = docs_data["docs"]
        context = DocumentFormatter.format_documents(docs_list)
        
        return {"question": query, "context": context}
    
    def recommend_cards(self, user_query: str) -> str:
        """카드 추천 실행"""
        if not self.rag_chain:
            raise ValueError("RAG 체인이 초기화되지 않았습니다.")
        
        return self.rag_chain.invoke(user_query)


# 싱글톤 인스턴스 관리
_recommendation_system: Optional[CardRecommendationSystem] = None

def get_recommendation_system() -> CardRecommendationSystem:
    """카드 추천 시스템 인스턴스 반환 (싱글톤 패턴)"""
    global _recommendation_system
    if _recommendation_system is None:
        _recommendation_system = CardRecommendationSystem()
        _recommendation_system.initialize()
    return _recommendation_system


 