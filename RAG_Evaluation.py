'''
RAGAS를 이용한 RAG 시스템 성능 평가
'''
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import random
import re
import time

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_similarity
from datasets import Dataset

# RecommendCard.py에서 카드 추천 시스템 import
from RecommendCard import get_recommendation_system

MAX_TOKENS = 512

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
def count_tokens(text):
    return len(tokenizer.encode(text))
def truncate_tokens(text, max_tokens=256):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

@dataclass  # RAGAS 평가 결과(점수 등) 저장 데이터 클래스
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    faithfulness_score: float
    answer_relevancy_score: float
    average_score: float
    interpretation: str
    raw_results: Any


class EnvironmentValidator:  # 환경 변수(OPENAI API KEY 등) 검증 클래스
    """환경 변수 검증 클래스"""
    
    @staticmethod
    def validate_openai_key() -> None:
        """OpenAI API 키 검증"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다.\n"
                "1. .env 파일에 OPENAI_API_KEY=your_api_key를 추가하세요.\n"
                "2. 또는 환경 변수로 설정하세요."
            )


class DatasetBuilder:  # 평가용 데이터셋 생성 및 컨텍스트 추출 클래스
    """평가용 데이터셋 생성 클래스"""
    
    def __init__(self, recommendation_system):
        self.recommendation_system = recommendation_system
    
    def create_single_question_dataset(self, user_question: str) -> Dataset:
        """단일 질문에 대한 평가용 데이터셋 생성"""
        # 카드 추천 시스템을 사용하여 답변 생성
        answer = self.recommendation_system.recommend_cards(user_question)
        
        # 컨텍스트 추출
        context_texts = self._extract_contexts(user_question)
        
        # RAGAS 요구사항에 맞는 데이터셋 생성
        # contexts는 각 질문에 대해 리스트 형태여야 함
        return Dataset.from_dict({
            "question": [user_question],
            "answer": [answer],
            "contexts": [context_texts]  # 각 질문에 대한 컨텍스트 리스트
        })
    
    def _extract_contexts(self, user_question: str) -> List[str]:
        """질문에 대한 컨텍스트 추출"""
        if not self.recommendation_system.model_manager.retriever:
            raise ValueError("Retriever가 초기화되지 않았습니다.")
        
        docs = self.recommendation_system.model_manager.retriever.invoke(user_question)
        context_texts = []
        
        for doc in docs:
            if hasattr(doc, 'page_content'):
                context_texts.append(doc.page_content)
        
        print(f"추출된 컨텍스트 수: {len(context_texts)}")
        if context_texts:
            print(f"첫 번째 컨텍스트 미리보기: {context_texts[0][:200]}...")
        
        return context_texts

    def create_synthetic_evaluation_set(self, n: int = 10) -> Dataset:
        db = self.recommendation_system.model_manager.db
        if not db:
            raise ValueError("FAISS DB가 초기화되지 않았습니다.")
        all_docs = db.similarity_search("", k=db.index.ntotal)
        filtered_docs = [doc for doc in all_docs if count_tokens(doc.page_content) <= 512]
        if len(filtered_docs) < n:
            raise ValueError(f"DB 내 적절한 길이의 문서가 부족합니다. (총 {len(filtered_docs)}개)")
        sampled_docs = random.sample(filtered_docs, n)
        questions = [truncate_tokens(doc.page_content, max_tokens=256) for doc in sampled_docs]
        answers = [self.recommendation_system.recommend_cards(q) for q in questions]
        contexts = [[truncate_tokens(doc.page_content, max_tokens=256)] for doc in sampled_docs]
        references = [truncate_tokens(doc.page_content, max_tokens=256) for doc in sampled_docs]
        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "reference": references
        })


class RAGASEvaluator:  # RAGAS 평가 실행 및 결과 파싱 클래스
    """RAGAS 평가 실행 클래스"""
    
    METRICS = [faithfulness, answer_relevancy, context_recall, context_precision, answer_similarity]
    
    @staticmethod
    def evaluate_dataset(dataset: Dataset):
        print("RAGAS 평가 실행 중...")
        results = evaluate(dataset, RAGASEvaluator.METRICS)
        print("\n[RAGAS 결과 객체 타입]", type(results))
        print("[RAGAS 결과 객체 dir]", dir(results))
        print("[RAGAS 결과 객체 내용]", results)
        if hasattr(results, 'scores'):
            print("[results.scores]", results.scores)
        if hasattr(results, '_scores_dict'):
            print("[results._scores_dict]", results._scores_dict)
        if isinstance(results, dict):
            print("[results dict keys]", results.keys())
        return results


def print_synthetic_evaluation_results(dataset, results, n):
    print("\n=== Synthetic 평가 결과 샘플 ===")
    for i in range(n):
        print(f"\n--- [Sample {i+1}] ---")
        print(f"[질문/문서] {dataset['question'][i][:200]}...")
        print(f"[답변] {dataset['answer'][i][:200]}...")
    print("\n==================== RAGAS 평가 지표 ====================")
    metrics = [
        ("faithfulness", "정합성 (faithfulness)"),
        ("answer_relevancy", "정답 관련성 (answer_relevancy)"),
        ("context_recall", "컨텍스트 재현율 (context_recall)"),
        ("context_precision", "컨텍스트 정밀도 (context_precision)"),
        ("semantic_similarity", "의미 유사도 (semantic_similarity)")
    ]
    scores_dict = getattr(results, '_scores_dict', None)
    if scores_dict is None and hasattr(results, 'scores'):
        scores_dict = {}
        for metric, _ in metrics:
            values = [d[metric] for d in results.scores if metric in d]
            if values:
                scores_dict[metric] = values
    for metric, label in metrics:
        value = None
        if scores_dict and metric in scores_dict:
            vals = scores_dict[metric]
            if isinstance(vals, list) and vals:
                value = sum(vals) / len(vals)
            else:
                value = vals
        print(f"{label:>20}: {value:.4f}" if value is not None else f"{label:>20}: N/A")
    print("========================================================\n")


# main 함수: 실행시간 측정 추가

def main():
    from RecommendCard import get_recommendation_system
    start = time.time()
    recommendation_system = get_recommendation_system()
    dataset_builder = DatasetBuilder(recommendation_system)
    n = 10
    dataset = dataset_builder.create_synthetic_evaluation_set(n)
    results = RAGASEvaluator.evaluate_dataset(dataset)
    print_synthetic_evaluation_results(dataset, results, n)
    end = time.time()
    print(f"\n총 실행 시간: {end - start:.2f}초")
    
    # 평가 결과 반환
    return results, dataset, end - start

def generate_report_from_results(results, dataset, execution_time):
    """평가 결과를 바탕으로 보고서 생성"""
    # 보고서 생성 여부 확인
    while True:
        print("\n" + "="*50)
        print("평가 보고서 생성")
        print("="*50)
        user_input = input("평가 결과를 PDF 보고서로 생성하시겠습니까? (Y/N): ").strip().upper()
        
        if user_input == 'Y':
            print("\n보고서 생성 중...")
            try:
                # 현재 작업 디렉토리 확인
                import os
                print(f"현재 작업 디렉토리: {os.getcwd()}")
                
                # RAG_Evaluation_Report.py 파일 존재 확인
                if not os.path.exists("RAG_Evaluation_Report.py"):
                    print("RAG_Evaluation_Report.py 파일을 찾을 수 없습니다.")
                    break
                
                print("RAG_Evaluation_Report.py 파일 발견")
                
                # 실제 평가 결과를 전달하여 보고서 생성
                import RAG_Evaluation_Report
                RAG_Evaluation_Report.create_report_with_results(results, dataset, execution_time)
                print("보고서 생성이 완료되었습니다!")
                
                # 생성된 파일 확인
                import glob
                reports_files = glob.glob("reports/*")
                if reports_files:
                    print("생성된 파일들:")
                    for file in reports_files:
                        print(f"  - {file}")
                else:
                    print("reports 폴더에 파일이 생성되지 않았습니다.")
                    
            except ImportError as e:
                print(f"모듈 import 오류: {e}")
                print("필요한 라이브러리가 설치되어 있는지 확인하세요:")
                print("pip install reportlab matplotlib")
            except Exception as e:
                print(f"보고서 생성 중 오류가 발생했습니다: {e}")
                import traceback
                traceback.print_exc()
            break
        elif user_input == 'N':
            print("보고서 생성을 건너뜁니다.")
            break
        else:
            print("잘못된 입력입니다. Y 또는 N을 입력해주세요.")


if __name__ == "__main__":
    results, dataset, execution_time = main()
    generate_report_from_results(results, dataset, execution_time) 