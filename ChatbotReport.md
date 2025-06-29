# 신용카드 챗봇 시스템

## 개요

본 시스템은 사용자의 소비 성향 및 원하는 혜택에 따라 최적의 신용카드를 추천하는 RAG 기반 챗봇입니다.  
데이터 수집부터 데이터 임베딩, 벡터 DB 저장, 벡터 검색, 프롬프트 생성, LLM 응답 출력에 이르는 과정을 LangChain으로 구성하고  
RAGAS 지표를 통해 성능 평가를 진행하여 챗봇의 성능을 검토했습니다.

---

## 시스템 구성

### 1) 데이터 수집 (크롤링)

카드 고릴라의 카드 정보 페이지를 `card_id` 기반으로 순차적으로 접근하여 신용카드의 정보를 크롤링합니다.  
`card_id`를 통한 URL이 있더라도 카드 정보가 없는 URL은 skip합니다.(약 85개)
카드의 발급 유무 정보를 담고 있는 class=`inactive`의 경우 발급이 불가한 경우만 확인할 수 있습니다.  
그러므로 class=`inactive`가 없는 경우 `inactive=False`로 부여하여 전처리를 진행했습니다.  
데이터 크롤링 시 이후에 진행할 데이터 임베딩을 위해 크롤링 후 JSON 형식으로 저장하게끔 했습니다.

---

### 2) 데이터 임베딩

카드 JSON 파일들을 Document 객체로 변환하여 다음 정보들을 분리하여 저장했습니다.

- 기본 카드 정보  
  (카드명, 카드사, 연회비, 브랜드(VISA, AMEX), 카드 발급 여부)

- 혜택 및 유의사항  
  (type, summary, details)

KoSimCSE (BM-K/KoSimCSE-roberta-multitask) 모델을 Hugging Face Hub에서 로딩하여 임베딩을 수행했습니다.  
문서 개수가 많고 길이도 다양하므로 chunking, batch_size 조절을 통해 약 17,000개의 문서를 임베딩했습니다.

---

### 3) 체인 구성

```python
RunnableLambda(self._retrieve_and_filter_docs)
| RunnableLambda(self._format_docs_for_prompt)
| PromptBuilder.create_recommendation_prompt()
| self.model_manager.llm
| StrOutputParser()
```

- 문서 검색 + 메타 필터링
- 문서 포맷팅
- 프롬프트 구성
- LLM 응답 생성

LangChain 기반의 Runnable 체인을 통해 사용자의 질문을 카드 정보와 연결하여 추천 응답을 생성합니다.

---

### 4) RAG 성능 평가

단일 질문에 대해 추천 결과를 생성한 후, 다음 두 지표에 기반한 평가를 수행합니다.

- faithfulness
- answer_relevancy

`ragas.evaluate`를 통해 평가를 진행하며, 평가 점수 해석 기준은 다음과 같습니다.

- 0.8 이상: 우수
- 0.6 이상: 양호
- 0.6 미만: 개선 필요