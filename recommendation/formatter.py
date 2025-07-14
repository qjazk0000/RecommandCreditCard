'''
문서 포맷팅 모듈 - 문서 → 프롬프트 입력 변환
'''
from typing import List


class DocumentFormatter:
    """문서 포맷팅 클래스 - 카드 문서(혜택 등)를 프롬프트용 문자열로 변환"""
    
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