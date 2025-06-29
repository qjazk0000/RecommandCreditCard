# 신용카드 챗봇 시스템

## 개요

## 프로젝트 소개

사용자의 소비 성향과 원하는 혜택에 따라 최적의 신용카드를 추천하는 RAG 기반 AI 챗봇입니다.  
질문 한 줄만으로 방대한 카드 혜택 정보를 요약하고 비교해, 빠르고 정확한 카드 선택을 돕습니다.

## 프로젝트 필요성

수많은 카드사와 혜택이 혼재된 시장에서 소비자는 자신에게 맞는 카드를 찾기 어려운 상황입니다.  
공식 홈페이지나 블로그, 광고 정보는 분산되어 있어 객관적인 비교가 어렵고, 소비자 입장에서 실질적인 정보 접근성도 낮습니다.  
특히 특정 혜택(예: 주유, 해외 결제, 스트리밍 할인 등)을 중점적으로 비교하고자 할 때, 신뢰할 수 있는 정보 제공 시스템의 부재가 문제가 되고 있습니다.

이에 따라, 신용카드 혜택 정보를 통합 수집하고 사용자의 질문에 맞춰 맞춤형으로 추천해주는 AI 기반 챗봇의 필요성이 대두되고 있습니다.

## 프로젝트 목표

- 카드 혜택, 연회비, 발급 조건 등 다양한 정보를 통합적으로 제공하는 신용카드 추천 챗봇을 구축합니다.  
- RAG(Retrieval-Augmented Generation) 구조를 기반으로, 문서 검색과 LLM 응답 생성을 결합하여 사용자 질문에 정확한 답변을 제공합니다.  
- LangChain 기반 파이프라인을 통해 크롤링, 전처리, 벡터화, 검색, 응답 생성을 체계화하고, RAGAS 지표를 활용해 성능을 정량적으로 평가합니다.  
- 사용자 신뢰도 확보를 위해 카드 상세 페이지 링크와 혜택 출처 정보를 함께 제공합니다.  
- 누구나 쉽게 접근 가능한 웹 인터페이스와 챗봇 구조를 통해 카드 정보 탐색의 진입 장벽을 낮추고, 소비자 선택을 돕습니다.


---

## 시스템 구성

### 1) 데이터 수집 (크롤링)

카드 고릴라의 카드 정보 페이지를 `card_id` 기반으로 순차적으로 접근하여 신용카드의 정보를 크롤링합니다.  
`card_id`를 통한 URL이 있더라도 카드 정보가 없는 URL은 skip합니다.(약 85개)  
카드의 발급 유무 정보를 담고 있는 class=`inactive`의 경우 발급이 불가한 경우만 확인할 수 있습니다.  
그러므로 class=`inactive`가 없는 경우 `inactive=False`로 부여하여 전처리를 진행했습니다.  
데이터 크롤링 시 이후에 진행할 데이터 임베딩을 위해 크롤링 후 JSON 형식으로 저장하게끔 했습니다.

다음은 크롤링된 카드 정보의 전처리 항목 정의입니다.

#### 카드 기본 정보

| 항목명                      | 설명                             | 데이터 타입     | 처리 방식 / 비고 |
|---------------------------|----------------------------------|----------------|------------------|
| `card_id`                 | 카드 고유 ID                      | Integer        | URL에서 추출, 카드 식별용 |
| `card_url`                | 카드 상세 페이지 URL              | String         | `https://www.card-gorilla.com/card/detail/{card_id}` 형식 |
| `card_name`               | 카드 이름                         | String         | HTML `.tit .card` 요소에서 추출 |
| `issuer`                 | 카드사 이름                        | String         | HTML `.brand` 클래스에서 추출 |
| `inactive`                | 발급 가능 여부                    | Boolean/String | `class="inactive"` 존재 시 True, 없으면 `False` 지정 |
| `brands`                  | 카드 브랜드 목록                   | List[String]   | `.c_brand span` 텍스트 리스트 |
| `annual_fee`              | 연회비 (국내/해외)                 | Dict           | `{domestic: ~, international: ~}` 구조 |
| `required_spending`       | 전월 실적 조건 설명                | String         | “전월 실적” 항목에서 텍스트 추출 |
| `required_spending_amount`| 실적 조건 금액 (숫자형)            | Integer/Null   | 텍스트에서 숫자만 정규표현식으로 추출 |

#### 혜택 정보

| 항목명              | 설명                     | 데이터 타입     | 처리 방식 / 비고 |
|-------------------|--------------------------|----------------|------------------|
| `benefits`        | 혜택 정보 전체 목록        | List[Object]   | 각 `dl` 블록에서 분리 추출 |
| `benefit.type`    | 혜택 분류 유형             | String         | `dt .txt1` 요소 |
| `benefit.summary` | 혜택 요약 설명             | String         | `dt i` 요소 |
| `benefit.details` | 혜택 상세 내용 리스트       | List[String]   | `dd > div > p` 텍스트 리스트 |

#### 유의사항

| 항목명     | 설명               | 데이터 타입     | 처리 방식 / 비고 |
|------------|--------------------|------------------|------------------|
| `cautions` | 유의사항 텍스트 목록 | List[String]     | “유의사항” 텍스트 포함 여부 기반 추출 |

<img width="1200px" src="./docs/crawling_save_json.svg" alt="신용카드 크롤링 흐름도">

---

### 2) 데이터 임베딩

카드 JSON 파일들을 Document 객체로 변환하여 다음 정보들을 분리하여 저장했습니다.

- 기본 카드 정보  
  (카드명, 카드사, 연회비, 브랜드(VISA, AMEX), 카드 발급 여부)

- 혜택 및 유의사항  
  (type, summary, details)

KoSimCSE (BM-K/KoSimCSE-roberta-multitask) 모델을 Hugging Face Hub에서 로딩하여 임베딩을 수행했습니다.  
문서 개수가 많고 길이도 다양하므로 chunking, batch_size 조절을 통해 약 17,000개의 문서를 임베딩했습니다.

<img width="1200px" src="./docs/create_FAISSDB.svg" alt="FAISS DB 생성 및 저장">

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

<img width="1200px" src="./docs/compose_RAG_chain.svg" alt="신용카드 추천 CHAIN 구성">

---

### 4) RAG 성능 평가

단일 질문에 대해 추천 결과를 생성한 후, 다음 두 지표에 기반한 평가를 수행합니다.

- faithfulness
- answer_relevancy

`ragas.evaluate`를 통해 평가를 진행하며, 평가 점수 해석 기준은 다음과 같습니다.

- 0.8 이상: 우수
- 0.6 이상: 양호
- 0.6 미만: 개선 필요

<img width="1200px" src="./docs/evaluate_RAGAS.svg" alt="신용카드 추천 챗봇 RAGAS 평가">

---
