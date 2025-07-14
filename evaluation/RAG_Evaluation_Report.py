import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import datetime

def get_korean_font():
    # 우선순위: 맑은고딕, 나눔고딕, Arial Unicode MS
    font_candidates = [
        ("MalgunGothic", "C:/Windows/Fonts/malgun.ttf"),
        ("NanumGothic", "C:/Windows/Fonts/NanumGothic.ttf"),
        ("ArialUnicodeMS", "C:/Windows/Fonts/arialuni.ttf"),
    ]
    for font_name, font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
            except Exception as e:
                continue
    return None

def extract_scores_from_results(results):
    """RAGAS 결과에서 점수 추출"""
    scores = {}
    
    # results._scores_dict에서 평균값 계산
    if hasattr(results, '_scores_dict') and results._scores_dict:
        for metric, values in results._scores_dict.items():
            if isinstance(values, list) and values:
                scores[metric] = sum(values) / len(values)
            else:
                scores[metric] = values
    # results.scores에서 평균값 계산 (fallback)
    elif hasattr(results, 'scores') and results.scores:
        metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'semantic_similarity']
        for metric in metrics:
            values = [d[metric] for d in results.scores if metric in d]
            if values:
                scores[metric] = sum(values) / len(values)
    
    return scores

def create_rag_evaluation_report(filename_suffix, scores=None, execution_time=None):
    """RAGAS 평가 결과를 PDF 보고서로 생성"""
    font_name = get_korean_font()
    if font_name is None:
        print("[경고] 한글 폰트를 찾을 수 없습니다. PDF는 영문만 정상 출력됩니다.")
        font_name = 'Helvetica'
    
    # 기본값 설정
    if scores is None:
        scores = {
            'faithfulness': 0.5156,
            'answer_relevancy': 0.6622,
            'context_recall': 1.0000,
            'context_precision': 1.0000,
            'semantic_similarity': 0.9026
        }
    
    if execution_time is None:
        execution_time = 203.0
    
    # reports 폴더 생성
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    pdf_filename = os.path.join(reports_dir, f"RAG_Evaluation_Report_{filename_suffix}.pdf")
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4, 
                          rightMargin=72, leftMargin=72, 
                          topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        fontName=font_name
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        fontName=font_name
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        fontName=font_name
    )
    story = []
    story.append(Paragraph("RAG 시스템 평가 보고서", title_style))
    story.append(Spacer(1, 20))
    current_date = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    story.append(Paragraph(f"생성일: {current_date}", normal_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph("1. 평가 개요", heading_style))
    story.append(Paragraph("• 평가 방법: RAGAS (RAG Assessment) 프레임워크", normal_style))
    story.append(Paragraph("• 평가 데이터: Synthetic 평가 세트 (10개 샘플)", normal_style))
    story.append(Paragraph("• 평가 모델: klue/roberta-base 토크나이저", normal_style))
    story.append(Paragraph(f"• 평가 시간: 약 {execution_time:.1f}초", normal_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("2. 평가 지표 설명", heading_style))
    metrics_explanation = [
        ["지표명", "설명", "점수 범위"],
        ["정합성 (Faithfulness)", "생성된 답변이 제공된 컨텍스트에 얼마나 충실한지 측정", "0~1 (높을수록 좋음)"],
        ["정답 관련성 (Answer Relevancy)", "생성된 답변이 사용자 질문과 얼마나 관련있는지 측정", "0~1 (높을수록 좋음)"],
        ["컨텍스트 재현율 (Context Recall)", "관련된 모든 정보가 컨텍스트에 포함되었는지 측정", "0~1 (높을수록 좋음)"],
        ["컨텍스트 정밀도 (Context Precision)", "컨텍스트에 포함된 정보가 얼마나 관련있는지 측정", "0~1 (높을수록 좋음)"],
        ["의미 유사도 (Semantic Similarity)", "생성된 답변과 참조 답변의 의미적 유사도 측정", "0~1 (높을수록 좋음)"]
    ]
    t = Table(metrics_explanation, colWidths=[2.2*inch, 3.3*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),
        ('WORDWRAP', (0, 0), (-1, -1), True),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    story.append(Paragraph("3. 평가 결과", heading_style))
    
    # 실제 점수로 등급 및 해석 결정
    def get_grade_and_interpretation(score):
        if score >= 0.8:
            return "우수", "매우 높은 성능"
        elif score >= 0.6:
            return "양호", "적당한 수준"
        else:
            return "보통", "개선 필요"
    
    results_data = [
        ["평가 지표", "점수", "등급", "해석"],
        ["정합성 (Faithfulness)", f"{scores.get('faithfulness', 0):.4f}", 
         *get_grade_and_interpretation(scores.get('faithfulness', 0))],
        ["정답 관련성 (Answer Relevancy)", f"{scores.get('answer_relevancy', 0):.4f}", 
         *get_grade_and_interpretation(scores.get('answer_relevancy', 0))],
        ["컨텍스트 재현율 (Context Recall)", f"{scores.get('context_recall', 0):.4f}", 
         *get_grade_and_interpretation(scores.get('context_recall', 0))],
        ["컨텍스트 정밀도 (Context Precision)", f"{scores.get('context_precision', 0):.4f}", 
         *get_grade_and_interpretation(scores.get('context_precision', 0))],
        ["의미 유사도 (Semantic Similarity)", f"{scores.get('semantic_similarity', 0):.4f}", 
         *get_grade_and_interpretation(scores.get('semantic_similarity', 0))]
    ]
    
    t2 = Table(results_data, colWidths=[2.2*inch, 1*inch, 1*inch, 2.3*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (3, 1), (3, -1), 'LEFT'),
        ('WORDWRAP', (0, 0), (-1, -1), True),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t2)
    story.append(Spacer(1, 20))
    story.append(Paragraph("4. 성능 분석", heading_style))
    analysis_text = [
        "• <b>우수한 부분:</b>",
        "  - 컨텍스트 재현율과 정밀도가 1.0으로 완벽함",
        "  - 의미 유사도가 0.90으로 매우 높음",
        "  - 정보 검색 및 임베딩 시스템이 효과적으로 작동",
        "",
        "• <b>개선이 필요한 부분:</b>",
        "  - 정합성(0.52)이 보통 수준으로, 답변 생성 품질 향상 필요",
        "  - 정답 관련성(0.66)이 양호하지만 더 개선 가능",
        "",
        "• <b>전체 평가:</b>",
        "  - 정보 검색 및 임베딩: 우수 (검색 엔진이 관련 정보를 잘 찾음)",
        "  - 답변 생성 품질: 보통~양호 (생성 모델의 정확성 개선 필요)",
        "  - 시스템 안정성: 우수 (일관된 성능 제공)"
    ]
    for line in analysis_text:
        if line.strip():
            story.append(Paragraph(line, normal_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("5. 개선 방안", heading_style))
    improvement_text = [
        "• <b>답변 생성 품질 향상:</b>",
        "  - 프롬프트 엔지니어링 개선",
        "  - 컨텍스트 길이 최적화",
        "  - 답변 검증 로직 추가",
        "",
        "• <b>모델 튜닝:</b>",
        "  - 더 정확한 답변 생성을 위한 파인튜닝",
        "  - 도메인 특화 학습 데이터 추가",
        "",
        "• <b>평가 지속:</b>",
        "  - 정기적인 성능 모니터링",
        "  - 사용자 피드백 반영"
    ]
    for line in improvement_text:
        if line.strip():
            story.append(Paragraph(line, normal_style))
    doc.build(story)
    print(f"RAG 평가 보고서가 '{pdf_filename}'로 생성되었습니다. (폰트: {font_name})")

def create_visualization(filename_suffix, scores=None):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    metrics = ['Faithfulness', 'Answer\nRelevancy', 'Context\nRecall', 'Context\nPrecision', 'Semantic\nSimilarity']
    
    # 실제 점수 사용 또는 기본값
    if scores is None:
        scores = {
            'faithfulness': 0.5156,
            'answer_relevancy': 0.6622,
            'context_recall': 1.0000,
            'context_precision': 1.0000,
            'semantic_similarity': 0.9026
        }
    
    score_values = [
        scores.get('faithfulness', 0),
        scores.get('answer_relevancy', 0),
        scores.get('context_recall', 0),
        scores.get('context_precision', 0),
        scores.get('semantic_similarity', 0)
    ]
    
    colors_list = []
    for score in score_values:
        if score >= 0.8:
            colors_list.append('#2E8B57')
        elif score >= 0.6:
            colors_list.append('#FFA500')
        else:
            colors_list.append('#FF6347')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    bars = ax1.bar(metrics, score_values, color=colors_list, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('RAG System Evaluation Results', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    for bar, score in zip(bars, score_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # 등급별 분포 계산
    excellent_count = sum(1 for score in score_values if score >= 0.8)
    good_count = sum(1 for score in score_values if 0.6 <= score < 0.8)
    needs_improvement_count = sum(1 for score in score_values if score < 0.6)
    
    categories = ['Excellent (0.8+)', 'Good (0.6-0.8)', 'Needs Improvement (<0.6)']
    counts = [excellent_count, good_count, needs_improvement_count]
    colors_pie = ['#2E8B57', '#FFA500', '#FF6347']
    
    wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.0f%%', 
                                       colors=colors_pie, startangle=90)
    ax2.set_title('Evaluation Metrics Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # reports 폴더 생성
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    chart_filename = os.path.join(reports_dir, f'RAG_Evaluation_Chart_{filename_suffix}.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"평가 결과 차트가 '{chart_filename}'로 저장되었습니다.")

def create_report_with_results(results, dataset, execution_time):
    """실제 평가 결과를 받아서 보고서 생성"""
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print("RAG 평가 보고서 생성 중...")
    
    # 결과에서 점수 추출
    scores = extract_scores_from_results(results)
    print(f"추출된 점수: {scores}")
    
    create_visualization(now, scores)
    create_rag_evaluation_report(now, scores, execution_time)
    print("\n보고서 생성 완료!")
    print(f"• reports/RAG_Evaluation_Report_{now}.pdf - 상세 평가 보고서")
    print(f"• reports/RAG_Evaluation_Chart_{now}.png - 시각화 차트")

if __name__ == "__main__":
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print("RAG 평가 보고서 생성 중...")
    create_visualization(now)
    create_rag_evaluation_report(now)
    print("\n보고서 생성 완료!")
    print(f"• reports/RAG_Evaluation_Report_{now}.pdf - 상세 평가 보고서")
    print(f"• reports/RAG_Evaluation_Chart_{now}.png - 시각화 차트") 