'''
RAGAS를 이용한 RAG 시스템 성능 평가
'''
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# RecommendCard.py에서 카드 추천 시스템 import
from RecommendCard import get_recommendation_system


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    faithfulness_score: float
    answer_relevancy_score: float
    average_score: float
    interpretation: str
    raw_results: Any


class EnvironmentValidator:
    """환경 변수 검증 클래스"""
    
    @staticmethod
    def validate_openai_key() -> None:
        """OpenAI API 키 검증"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "❌ OpenAI API 키가 설정되지 않았습니다.\n"
                "1. .env 파일에 OPENAI_API_KEY=your_api_key를 추가하세요.\n"
                "2. 또는 환경 변수로 설정하세요."
            )


class DatasetBuilder:
    """평가용 데이터셋 생성 클래스"""
    
    def __init__(self, recommendation_system):
        self.recommendation_system = recommendation_system
    
    def create_single_question_dataset(self, user_question: str) -> Dataset:
        """단일 질문에 대한 평가용 데이터셋 생성"""
        # 카드 추천 시스템을 사용하여 답변 생성
        answer = self.recommendation_system.recommend_cards(user_question)
        
        # 컨텍스트 추출
        context_texts = self._extract_contexts(user_question)
        
        # 데이터셋 생성
        return Dataset.from_dict({
            "question": [user_question],
            "answer": [answer],
            "contexts": [context_texts]
        })
    
    def _extract_contexts(self, user_question: str) -> List[str]:
        """질문에 대한 컨텍스트 추출"""
        if not self.recommendation_system.retriever:
            raise ValueError("Retriever가 초기화되지 않았습니다.")
        
        docs = self.recommendation_system.retriever.invoke(user_question)
        context_texts = []
        
        for doc in docs:
            if hasattr(doc, 'page_content'):
                context_texts.append(doc.page_content)
        
        return context_texts


class RAGASEvaluator:
    """RAGAS 평가 실행 클래스"""
    
    METRICS = [faithfulness, answer_relevancy]
    
    @staticmethod
    def evaluate_dataset(dataset: Dataset) -> EvaluationResult:
        """데이터셋에 대한 RAGAS 평가 실행"""
        print("RAGAS 평가 실행 중...")
        
        # 평가 실행
        results = evaluate(dataset, RAGASEvaluator.METRICS)
        
        # 결과 파싱
        return RAGASEvaluator._parse_results(results)
    
    @staticmethod
    def _parse_results(results: Any) -> EvaluationResult:
        """RAGAS 결과 파싱"""
        try:
            # 결과를 딕셔너리로 변환
            results_dict = {}
            for metric in RAGASEvaluator.METRICS:
                metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
                
                if hasattr(results, metric_name):
                    results_dict[metric_name] = getattr(results, metric_name)
                elif isinstance(results, dict) and metric_name in results:
                    results_dict[metric_name] = results[metric_name]
            
            # 점수 추출
            faithfulness_score = results_dict.get('faithfulness', 0.0)
            answer_relevancy_score = results_dict.get('answer_relevancy', 0.0)
            
            # 평균 점수 계산
            scores = [faithfulness_score, answer_relevancy_score]
            average_score = sum(scores) / len(scores)
            
            # 결과 해석
            interpretation = RAGASEvaluator._interpret_score(average_score)
            
            return EvaluationResult(
                faithfulness_score=faithfulness_score,
                answer_relevancy_score=answer_relevancy_score,
                average_score=average_score,
                interpretation=interpretation,
                raw_results=results
            )
            
        except Exception as e:
            print(f"결과 처리 중 오류: {e}")
            print("원본 결과:", results)
            
            # 기본값 반환
            return EvaluationResult(
                faithfulness_score=0.0,
                answer_relevancy_score=0.0,
                average_score=0.0,
                interpretation="결과 처리 실패",
                raw_results=results
            )
    
    @staticmethod
    def _interpret_score(score: float) -> str:
        """점수 해석"""
        if score >= 0.8:
            return "🟢 우수한 성능: 답변이 매우 정확하고 관련성이 높습니다."
        elif score >= 0.6:
            return "🟡 양호한 성능: 답변이 적절한 수준입니다."
        else:
            return "🔴 개선 필요: 답변의 품질을 향상시킬 필요가 있습니다."


class ResultPrinter:
    """결과 출력 클래스"""
    
    @staticmethod
    def print_evaluation_results(question: str, result: EvaluationResult, answer: str) -> None:
        """평가 결과 출력"""
        print(f"\n=== RAGAS 평가 결과 ===")
        print(f"faithfulness: {result.faithfulness_score:.3f}")
        print(f"answer_relevancy: {result.answer_relevancy_score:.3f}")
        print(f"\n평균 점수: {result.average_score:.3f}")
        
        print(f"\n=== 결과 해석 ===")
        print(result.interpretation)
        
        print(f"\n=== 생성된 답변 ===")
        print(answer)


class RAGEvaluator:
    """RAG 시스템 성능 평가 메인 클래스"""
    
    def __init__(self):
        self.recommendation_system = None
        self.dataset_builder = None
    
    def initialize(self) -> None:
        """시스템 초기화"""
        print("카드 추천 시스템 초기화 중...")
        
        # 환경 변수 검증
        load_dotenv()
        EnvironmentValidator.validate_openai_key()
        
        # 카드 추천 시스템 초기화
        self.recommendation_system = get_recommendation_system()
        self.dataset_builder = DatasetBuilder(self.recommendation_system)
        
        print("모델 초기화 완료")
    
    def evaluate_single_question(self, user_question: str) -> Optional[EvaluationResult]:
        """사용자 질문에 대한 단일 평가"""
        print(f"=== 질문 평가: {user_question} ===")
        
        try:
            if not self.dataset_builder:
                raise ValueError("데이터셋 빌더가 초기화되지 않았습니다.")
            
            # 데이터셋 생성
            dataset = self.dataset_builder.create_single_question_dataset(user_question)
            
            # RAGAS 평가 실행
            result = RAGASEvaluator.evaluate_dataset(dataset)
            
            # 답변 추출 (데이터셋에서)
            answer = dataset["answer"][0]
            
            # 결과 출력
            ResultPrinter.print_evaluation_results(user_question, result, answer)
            
            return result
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return None


class InteractiveEvaluator:
    """대화형 평가 인터페이스"""
    
    def __init__(self):
        self.evaluator = RAGEvaluator()
    
    def run(self) -> None:
        """대화형 평가 실행"""
        print("=== RAGAS 실시간 평가 시스템 ===")
        print("사용자 질문에 대해 RAGAS 지표를 실시간으로 계산합니다.")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print()
        
        try:
            self.evaluator.initialize()
            
            while True:
                user_question = self._get_user_input()
                
                if user_question is None:  # 종료 신호
                    break
                
                if not user_question:  # 빈 입력
                    print("질문을 입력해주세요.")
                    continue
                
                # 평가 실행
                self.evaluator.evaluate_single_question(user_question)
                print("\n" + "="*50 + "\n")
        
        except Exception as e:
            print(f"시스템 초기화 실패: {e}")
    
    def _get_user_input(self) -> Optional[str]:
        """사용자 입력 받기"""
        try:
            user_input = input("질문을 입력하세요: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("평가 시스템을 종료합니다.")
                return None
            
            return user_input
            
        except KeyboardInterrupt:
            print("\n평가 시스템을 종료합니다.")
            return None
        except EOFError:
            print("\n평가 시스템을 종료합니다.")
            return None


def main():
    """메인 실행 함수"""
    evaluator = InteractiveEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main() 