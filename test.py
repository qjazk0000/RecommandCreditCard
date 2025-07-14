'''
카드 추천 시스템 CLI 인터페이스
'''
from recommendation.recommender import get_recommendation_system


def main():
    """카드 추천 시스템 CLI 인터페이스"""
    print("=== 카드 추천 시스템 ===")
    print("사용자 입력에 따라 벡터 DB에서 유사도 높은 5개 카드를 검색하여")
    print("LLM이 최적의 3개 신용카드를 추천해드립니다.")
    print()
    print("필터링 옵션:")
    print("- 카드사 필터링 (예: '신한카드 중에서 외식 혜택 좋은 카드')")
    print("- 카드명 필터링 (예: '삼성 iD ON 카드 추천')")
    print("- 연회비 필터링 (예: '연회비 5만원 이하 카드 추천')")
    print("- 브랜드 필터링 (예: 'VISA 브랜드 카드 중에서 온라인 쇼핑 혜택 좋은 카드')")
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