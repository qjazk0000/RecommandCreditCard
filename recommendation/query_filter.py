'''
쿼리 필터링 모듈 - 사용자 쿼리에서 필터 추출 및 문서 필터링
'''
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class FilterConditions:
    """필터링 조건 데이터 클래스"""
    issuer: Optional[str] = None
    card_name: Optional[str] = None
    annual_fee: Optional[str] = None
    brands: Optional[str] = None


class QueryFilter:
    """쿼리 필터링 클래스 - 쿼리에서 필터 조건 추출 및 문서 필터링 담당"""
    
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