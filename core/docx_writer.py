# core/docx_writer.py
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from datetime import datetime
import os


def add_table_borders(table):
    """표에 테두리 추가"""
    tbl = table._element
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    
    tblBorders = OxmlElement('w:tblBorders')
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), '000000')
        tblBorders.append(border)
    tblPr.append(tblBorders)


def set_cell_background(cell, color):
    """셀 배경색 설정"""
    shading_elm = OxmlElement('w:shd')
    shading_elm.set(qn('w:fill'), color)
    cell._element.get_or_add_tcPr().append(shading_elm)


def parse_user_query(user_query: str) -> dict:
    """
    사용자 쿼리에서 공사명 / 사고발생장소 / 사고종류 / 사고개요 추출
    """
    data = {
        "공사명": "",
        "사고발생장소": "",
        "사고종류": "",
        "사고개요": ""
    }
    
    if not user_query:
        return data
    
    lines = user_query.strip().split('\n')
    for line in lines:
        line = line.strip()
        if '공종:' in line:
            data["공사명"] = line.split(':', 1)[1].strip()
        elif '작업프로세스' in line:
            data["사고발생장소"] = line.split(':', 1)[1].strip()
        elif '사고 유형' in line or '사고유형' in line:
            data["사고종류"] = line.split(':', 1)[1].strip()
        elif '사고 개요' in line:
            data["사고개요"] = line.split(':', 1)[1].strip()
    
    return data


def _fill_multiline_text(cell, text: str, font_size: int = 9):
    """
    셀에 여러 줄 텍스트(여러 문단)를 채워 넣기 위한 유틸.
    기존 내용 초기화 후 줄마다 새 paragraph 생성.
    """
    cell.text = ""
    if not text:
        return
    
    lines = text.split('\n')
    for line in lines:
        para = cell.add_paragraph(line.strip())
        para.paragraph_format.line_spacing = 1.3
        para.paragraph_format.space_after = Pt(4)
        for run in para.runs:
            run.font.size = Pt(font_size)


def create_accident_report_docx(
    user_query: str,
    cause_text: str,
    action_text: str,
    output_path: str = None,
    source_references: list = None,  # ✅ 추가
) -> str:
    """
    [별지 제2호 서식] 건설사고 발생현황 보고 양식 DOCX 생성
    + 관련 근거 자료 추가 (선택적)
    """
    doc = Document()
    
    # ==== 페이지 여백 ====
    sections = doc.sections
    for section in sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
    
    # ==== 헤더 ====
    header_para = doc.add_paragraph()
    header_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header_para.add_run('[별지 제2호 서식] 건설사고 발생현황 보고 양식')
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(100, 100, 100)
    
    # ==== 제목 ====
    title = doc.add_heading('건설사고 발생현황 보고', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.runs[0]
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.name = '맑은 고딕'
    
    doc.add_paragraph()
    
    # ==== 사용자 질의 파싱 ====
    query_data = parse_user_query(user_query)
    
    # ==== 기본정보 테이블 ====
    table1 = doc.add_table(rows=3, cols=4)
    table1.style = 'Table Grid'
    add_table_borders(table1)
    
    # 1행
    cells = table1.rows[0].cells
    set_cell_background(cells[0], 'E7E6E6')
    set_cell_background(cells[2], 'E7E6E6')
    cells[0].text = '수신'
    cells[1].text = ''
    cells[2].text = '보고일시'
    cells[3].text = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
    
    for cell in [cells[0], cells[2]]:
        for para in cell.paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.font.bold = True
                run.font.size = Pt(10)
    
    # 2행
    cells = table1.rows[1].cells
    set_cell_background(cells[0], 'E7E6E6')
    set_cell_background(cells[2], 'E7E6E6')
    cells[0].text = '발신기관'
    cells[1].text = ''
    cells[2].text = '발신(보고자)'
    cells[3].text = 'O O O (인)'
    
    # 3행
    cells = table1.rows[2].cells
    set_cell_background(cells[0], 'E7E6E6')
    cells[0].text = '제목'
    cells[1].merge(cells[2]).merge(cells[3])
    cells[1].text = '건설사고 발생현황 보고'
    
    doc.add_paragraph()
    
    # ==== 사고 상세 정보 테이블 ====
    table2 = doc.add_table(rows=15, cols=4)  # ✅ 15행으로 변경 (14행까지 사용 + 비고 1행)
    table2.style = 'Table Grid'
    add_table_borders(table2)
    
    row_data = [
        (0, '사고일시', datetime.now().strftime('%Y년 %m월 %d일 ( )요일  시  분 경'), [(2, '기상상태', '')]),
        (1, '공사명', query_data.get('공사명', ''), None),
        (2, '시공사', '책임자 및 연락처', None),
        (3, '건설사업관리기술자', '책임자 및 연락처', None),
        (4, '설계자', '책임자 및 연락처', None),
        (5, '현장주소', '', [(2, '사고발생장소', query_data.get('사고발생장소', ''))]),
        (6, '사고 종류', query_data.get('사고종류', ''), None),
        (7, '인적피해', '', [(2, '장비손실', '')]),
        (8, '구조물 손실', '', [(2, '피해금액', '')]),
        (9, '공기지연', '', [(2, '안전관리계획서\n수립여부', '해당 : (  ), 해당없음 : (  )\n해당사유 : 의 제98조제1항(  )호')]),
    ]
    
    for row_idx, label, value, extra in row_data:
        cells = table2.rows[row_idx].cells
        
        set_cell_background(cells[0], 'E7E6E6')
        cells[0].text = label
        cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cells[0].paragraphs[0].runs:
            run.font.bold = True
            run.font.size = Pt(9)
        
        if extra:
            cells[1].text = value
            set_cell_background(cells[2], 'E7E6E6')
            cells[2].text = extra[0][1]
            for run in cells[2].paragraphs[0].runs:
                run.font.bold = True
                run.font.size = Pt(9)
            cells[3].text = extra[0][2]
        else:
            cells[1].merge(cells[2]).merge(cells[3])
            cells[1].text = value
    
    # ==== 사고발생 경위 ====
    row_idx = 10
    cells = table2.rows[row_idx].cells
    set_cell_background(cells[0], 'E7E6E6')
    cells[0].text = '사고발생 경위\n(발생원인)'
    _fill_multiline_text(cells[1].merge(cells[2]).merge(cells[3]), cause_text)
    
    # ==== 조치사항 및 향후조치계획 ====
    row_idx = 11
    cells = table2.rows[row_idx].cells
    set_cell_background(cells[0], 'E7E6E6')
    cells[0].text = '조치사항 및\n향후조치계획'
    _fill_multiline_text(cells[1].merge(cells[2]).merge(cells[3]), action_text)
    
    # ==== 사고조사 방법 ====
    row_idx = 12
    cells = table2.rows[row_idx].cells
    set_cell_background(cells[0], 'E7E6E6')
    cells[0].text = '사고조사 방법'
    cells[1].merge(cells[2]).merge(cells[3]).text = "1. 직접조사\n2. 사고조사위원회조사\n3. 노동부 재해조사시 합동조사"
    
    # ✅ ==== 관련 근거 자료 (13행) ====
    row_idx = 13
    if source_references and len(source_references) > 0:
        cells = table2.rows[row_idx].cells
        set_cell_background(cells[0], 'E7E6E6')
        cells[0].text = '관련\n근거자료'
        cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cells[0].paragraphs[0].runs:
            run.font.bold = True
            run.font.size = Pt(9)
        
        # 근거 자료 내용 작성
        merged_cell = cells[1].merge(cells[2]).merge(cells[3])
        merged_cell.text = ""  # 초기화
        
        # 각 근거 자료를 문단으로 추가
        for i, ref in enumerate(source_references, 1):
            # 구분선 (첫 번째 제외)
            if i > 1:
                sep_para = merged_cell.add_paragraph()
                sep_run = sep_para.add_run('─' * 60)
                sep_run.font.size = Pt(8)
                sep_run.font.color.rgb = RGBColor(180, 180, 180)
            
            # 문서 제목
            title_para = merged_cell.add_paragraph()
            title_run = title_para.add_run(f'[문서 {ref["idx"]}] {ref["filename"]}')
            title_run.font.size = Pt(9)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(0, 70, 140)
            title_para.paragraph_format.space_after = Pt(2)
            
            # 위치 정보
            if ref.get("hierarchy"):
                loc_para = merged_cell.add_paragraph()
                loc_run1 = loc_para.add_run('📍 위치: ')
                loc_run1.font.size = Pt(8)
                loc_run1.font.bold = True
                loc_run2 = loc_para.add_run(ref["hierarchy"])
                loc_run2.font.size = Pt(8)
                loc_para.paragraph_format.space_after = Pt(2)
            elif ref.get("section"):
                loc_para = merged_cell.add_paragraph()
                loc_run1 = loc_para.add_run('📍 섹션: ')
                loc_run1.font.size = Pt(8)
                loc_run1.font.bold = True
                loc_run2 = loc_para.add_run(ref["section"])
                loc_run2.font.size = Pt(8)
                loc_para.paragraph_format.space_after = Pt(2)
            
            # 관련성 요약
            if ref.get("relevance_summary"):
                rel_para = merged_cell.add_paragraph()
                rel_run1 = rel_para.add_run('💡 관련성: ')
                rel_run1.font.size = Pt(8)
                rel_run1.font.bold = True
                rel_run2 = rel_para.add_run(ref["relevance_summary"])
                rel_run2.font.size = Pt(8)
                rel_para.paragraph_format.space_after = Pt(2)
            
            # 핵심 내용
            key_sentences = ref.get("key_sentences", [])
            if key_sentences:
                key_para = merged_cell.add_paragraph()
                key_run = key_para.add_run('🎯 핵심 내용:')
                key_run.font.size = Pt(8)
                key_run.font.bold = True
                key_para.paragraph_format.space_after = Pt(2)
                
                for sentence in key_sentences[:3]:  # 최대 3개만
                    sent_para = merged_cell.add_paragraph()
                    sent_para.paragraph_format.left_indent = Cm(0.5)
                    sent_run = sent_para.add_run(f'• {sentence}')
                    sent_run.font.size = Pt(8)
                    sent_para.paragraph_format.space_after = Pt(1)
    else:
        # 근거 자료가 없으면 비고로 사용
        cells = table2.rows[row_idx].cells
        set_cell_background(cells[0], 'E7E6E6')
        cells[0].text = '비고'
        cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cells[0].paragraphs[0].runs:
            run.font.bold = True
            run.font.size = Pt(9)
        cells[1].merge(cells[2]).merge(cells[3]).text = ""
    
    # ✅ ==== 비고 (14행, 맨 마지막) ====
    row_idx = 14
    cells = table2.rows[row_idx].cells
    set_cell_background(cells[0], 'E7E6E6')
    cells[0].text = '비고'
    cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cells[0].paragraphs[0].runs:
        run.font.bold = True
        run.font.size = Pt(9)
    cells[1].merge(cells[2]).merge(cells[3]).text = ""
    
    # ==== 파일 저장 ====
    if not os.path.exists("reports"):
        os.makedirs("reports")
    
    if output_path is None:
        output_path = f"reports/건설사고_발생현황_보고_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    
    doc.save(output_path)
    return output_path