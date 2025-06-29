'''
카드 추천 시스템 - RAG 기반 신용카드 추천
'''
import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from textwrap import dedent

import torch


@dataclass  # 카드 필터링 조건을 저장하는 데이터 클래스
class FilterConditions:
    """필터링 조건 데이터 클래스"""
    issuer: Optional[str] = None
    card_name: Optional[str] = None
    annual_fee: Optional[str] = None
    brands: Optional[str] = None


class ModelManager:  # 임베딩/LLM/DB 등 모델 및 벡터스토어 관리 클래스
    """모델 관리 클래스"""
    
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
        persist_dir = script_dir / "faiss_card_db"
        
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


class QueryFilter:  # 쿼리에서 필터 조건 추출 및 문서 필터링 담당 클래스
    """쿼리 필터링 클래스"""
    
    SUPPORTED_ISSUERS = [
        "신한", "삼성", "KB", "국민", "롯데", "현대", "하나", 
        "우리", "NH", "농협", "IBK", "기업은행", "BNK", "경남은행", 
        "부산은행", "DB", "금융투자", "iM", "뱅크", "MG", "새마을금고",
        "SBI", "저축은행", "SC", "제일은행", "Sh", "수협은행", 
        "SK", "증권", "SSG", "PAY", "광주은행", "교보증권", 
        "네이버페이", "다날", "머니트리", "미래에셋증권", "삼성증권",
        "신협", "씨티", "아이오로라", "엔에이치엔페이코", "우체국",
        "유안타증권", "유진투자증권", "전북은행", "제주은행", "차이",
        "카카오뱅크", "카카오페이", "케이뱅크", "코나카드", "토스",
        "토스뱅크", "트래블월렛", "핀크카드", "핀트", "한국투자증권",
        "한패스", "현대백화점"
    ]
    
    SUPPORTED_BRANDS = ["VISA", "MASTER", "JCB", "AMEX"]
    
    @staticmethod
    def extract_filter_conditions(query: str) -> FilterConditions:
        """쿼리에서 필터링 조건 추출"""
        conditions = FilterConditions()
        
        # 카드사 필터링
        for issuer in QueryFilter.SUPPORTED_ISSUERS:
            if issuer in query:
                conditions.issuer = issuer
                break
        
        # 카드명 필터링
        card_name_match = re.search(r'["\']([^"\']+)["\']', query)
        if card_name_match:
            conditions.card_name = card_name_match.group(1)
        
        # 연회비 필터링
        fee_match = re.search(r'연회비\s*(\d+)', query)
        if fee_match:
            conditions.annual_fee = fee_match.group(1)
        
        # 브랜드 필터링
        for brand in QueryFilter.SUPPORTED_BRANDS:
            if brand in query.upper():
                conditions.brands = brand
                break
        
        return conditions
    
    @staticmethod
    def apply_filters(docs_list: List, conditions: FilterConditions) -> List:
        """메타데이터 기반 필터링 적용"""
        if not conditions.issuer and not conditions.card_name and \
           not conditions.annual_fee and not conditions.brands:
            return docs_list
        
        filtered_docs = []
        for doc in docs_list:
            if not hasattr(doc, 'metadata'):
                continue
                
            metadata = doc.metadata
            if QueryFilter._matches_conditions(metadata, conditions):
                filtered_docs.append(doc)
        
        return filtered_docs if filtered_docs else docs_list
    
    @staticmethod
    def _matches_conditions(metadata: Dict, conditions: FilterConditions) -> bool:
        """메타데이터가 조건과 일치하는지 확인"""
        if conditions.issuer and conditions.issuer not in metadata.get("issuer", ""):
            return False
        
        if conditions.card_name and conditions.card_name not in metadata.get("card_name", ""):
            return False
        
        if conditions.annual_fee and conditions.annual_fee not in str(metadata.get("annual_fee", "")):
            return False
        
        if conditions.brands and conditions.brands not in metadata.get("brands", ""):
            return False
        
        return True


class DocumentFormatter:  # 카드 문서(혜택 등)를 프롬프트용 문자열로 변환하는 클래스
    """문서 포맷팅 클래스"""
    
    @staticmethod
    def format_documents(docs_list: List) -> str:
        """문서 리스트를 포맷팅된 문자열로 변환"""
        if not isinstance(docs_list, list) or not docs_list:
            return "관련 카드 정보를 찾을 수 없습니다."
        
        formatted_docs = []
        for doc in docs_list:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                formatted_docs.append(DocumentFormatter._format_single_document(doc))
            else:
                formatted_docs.append(str(doc))
        
        return "\n\n---\n\n".join(formatted_docs)
    
    @staticmethod
    def _format_single_document(doc) -> str:
        """단일 문서 포맷팅"""
        metadata = doc.metadata
        card_info = f"[카드사: {metadata.get('issuer', 'N/A')}] "
        card_info += f"[카드명: {metadata.get('card_name', 'N/A')}] "
        
        # URL을 더 명확하게 표시
        if metadata.get('card_url'):
            card_info += f"\n[카드 URL: {metadata.get('card_url')}] "
        
        return f"{card_info}\n{doc.page_content}"


class PromptBuilder:  # LLM 프롬프트 템플릿 생성 클래스
    """프롬프트 빌더 클래스"""
    
    @staticmethod
    def create_recommendation_prompt() -> ChatPromptTemplate:
        """카드 추천용 프롬프트 템플릿 생성"""
        return ChatPromptTemplate.from_messages([
            ("system", PromptBuilder._get_system_prompt()),
            ("human", "사용자 질문: {question}\n\n카드정보:\n{context}")
        ])
    
    @staticmethod
    def _get_system_prompt() -> str:
        """시스템 프롬프트 텍스트"""
        return dedent("""
        당신은 개인의 소비 성향과 필요에 따라 최적의 신용카드를 추천해주는 AI 챗봇입니다.

        당신에게는 다음 두 가지 정보가 주어집니다:
        1. 사용자의 질문 또는 소비 패턴 설명 (예: "주유 혜택 많은 카드 추천", "외식과 편의점 자주 씀")
        2. 카드별 혜택과 유의사항이 담긴 카드 정보 목록 (벡터 DB에서 유사도 높은 5개 카드)

        제공된 카드 정보를 바탕으로 사용자의 요구에 가장 잘 부합하는 **정확히 3개의 신용카드를 추천**해 주세요.

        **추천 규칙:**
        - 반드시 3개의 카드만 추천하세요
        - 사용자의 소비 패턴과 가장 관련성이 높은 카드를 우선적으로 추천하세요
        - 각 카드는 서로 다른 특징을 가져야 합니다 (다양성 확보)
        - 카드 정보에 없는 내용은 생성하지 마세요

        **각 카드별 필수 포함 정보:**
        - **카드사명**: 정확한 카드사 이름
        - **카드명**: 정확한 카드 이름
        - **카드 URL**: 반드시 포함 (카드 정보에 URL이 있다면 반드시 표시)
        
        **각 카드별 추천 내용:**
        - 해당 카드가 사용자의 소비 패턴과 어떻게 잘 맞는지 **명확한 이유를 들어 설명**
        - **주요 혜택을 항목별로 정리** (각 혜택을 별도 줄에 • 또는 - 기호로 구분)
        - 유의사항 요약
        - **5~8줄 이내로 충분히 상세하게 정리**

        **출력 형식:**
        ## 추천 카드 1
        **카드사**: [카드사명]
        **카드명**: [카드명]
        **카드 URL**: [URL] (URL이 있는 경우)
        
        [추천 이유]
        
        **주요 혜택:**
        • [혜택1]: [상세 내용]
        • [혜택2]: [상세 내용]
        • [혜택3]: [상세 내용]
        
        **유의사항:**
        • [유의사항1]
        • [유의사항2]

        ## 추천 카드 2
        **카드사**: [카드사명]
        **카드명**: [카드명]
        **카드 URL**: [URL] (URL이 있는 경우)
        
        [추천 이유]
        
        **주요 혜택:**
        • [혜택1]: [상세 내용]
        • [혜택2]: [상세 내용]
        • [혜택3]: [상세 내용]
        
        **유의사항:**
        • [유의사항1]
        • [유의사항2]

        ## 추천 카드 3
        **카드사**: [카드사명]
        **카드명**: [카드명]
        **카드 URL**: [URL] (URL이 있는 경우)
        
        [추천 이유]
        
        **주요 혜택:**
        • [혜택1]: [상세 내용]
        • [혜택2]: [상세 내용]
        • [혜택3]: [상세 내용]
        
        **유의사항:**
        • [유의사항1]
        • [유의사항2]

        **중요**: 
        - 반드시 3개의 카드만 추천하세요
        - 카드 정보에 URL이 포함되어 있다면 반드시 출력에 포함시켜주세요
        - 카드 정보에 없는 내용은 생성하지 마세요
        - 각 카드는 서로 다른 특징을 가져야 합니다
        """)


class CardRecommendationSystem:  # 카드 추천 시스템 전체 RAG 체인 관리 클래스
    """카드 추천 시스템 메인 클래스"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.rag_chain = None
    
    def initialize(self) -> None:
        """시스템 초기화"""
        print("모델 초기화 중...")
        
        # 환경 변수 로드
        load_dotenv()
        self._validate_environment()
        
        # 모델 초기화
        self.model_manager.initialize_embedding_model()
        self.model_manager.initialize_llm()
        self.model_manager.load_faiss_database()
        
        # RAG 체인 구성
        self._build_rag_chain()
        
        print("모델 초기화 완료")
    
    def _validate_environment(self) -> None:
        """환경 변수 검증"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다.\n"
                "1. .env 파일에 OPENAI_API_KEY=your_api_key를 추가하세요.\n"
                "2. 또는 환경 변수로 설정하세요."
            )
    
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


def main():  # 카드 추천 시스템 CLI 실행 함수
    """메인 실행 함수"""
    print("=== 카드 추천 시스템 ===")
    print("사용자 입력에 따라 벡터 DB에서 유사도 높은 5개 카드를 검색하여")
    print("LLM이 최적의 3개 신용카드를 추천해드립니다.")
    print()
    print("필터링 옵션:")
    print("- 카드사 필터링")
    print("- 카드명 필터링")
    print("- 연회비 필터링")
    print("- 브랜드 필터링")
    print()
    
    try:
        system = get_recommendation_system()
        
        while True:
            user_query = input("사용자 질문을 입력하세요 (종료: quit): ").strip()
            
            if user_query.lower() in ['quit', 'exit', '종료']:
                print("시스템을 종료합니다.")
                break
            
            if not user_query:
                print("질문을 입력해주세요.")
                continue
            
            try:
                print("\n벡터 DB에서 유사도 높은 5개 카드를 검색 중...")
                result = system.recommend_cards(user_query)
                print("\n=== 추천 결과 (3개 카드) ===\n")
                print(result)
                print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"추천 중 오류 발생: {e}")
                print()
    
    except Exception as e:
        print(f"시스템 초기화 실패: {e}")


if __name__ == "__main__":
    main()