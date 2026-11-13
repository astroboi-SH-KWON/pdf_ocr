import os
# --- [0xC0000005 에러 해결용 환경 변수 강제 설정] ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
# --------------------------------------------------

# import fitz  # PyMuPDF
import pymupdf
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


def create_searchable_pdf_with_paddle(input_pdf_path, output_pdf_path, dpi=300):
    """
    스캔된 PDF를 고해상도 이미지로 변환한 뒤, PaddleOCR로 텍스트와 좌표를 추출하여 원본 PDF 위에 투명 텍스트 레이어 올림.

    :param input_pdf_path: 스캔 PDF 경로
    :param output_pdf_path: OCR 결과가 포함된 출력 PDF 경로
    :param dpi: PDF 렌더링 해상도 (300 권장, 높을수록 인식률 상승 및 속도 저하)
    """
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_pdf_path}")

    # 1. PaddleOCR 엔진 초기화
    # lang='korean' 설정 시 한국어 + 영어 + 숫자 + 기본 기호를 모두 인식
    # 최초 실행 시 한국어 인식 모델(.tar)을 자동으로 다운로드.
    print("PaddleOCR 한국어 모델 로드 중...")
    ocr = PaddleOCR(
        # use_textline_orientation=True,  # 90도/180도 회전되거나 기울어진 글자 자동 보정
        use_textline_orientation=False,  # 회전 보정 끄기 (메모리 절약)
        lang="korean",       # 한국어 특화 모델 (영어 동시 지원)
        enable_mkldnn=False,  # OneDNN 충돌 버그 우회
        # use_mp=False,  # 멀티프로세싱 충돌 방지... 이런 param 없는데...
        # show_log=False       # 불필요한 디버그 로그 숨김... 이런 param 없는데...
    )

    doc = pymupdf.open(input_pdf_path)
    total_pages = len(doc)
    scale = dpi / 72.0  # PDF 기본 해상도(72 DPI) 대비 확대 비율
    matrix = pymupdf.Matrix(scale, scale)

    print(f"총 {total_pages}페이지 OCR 처리 시작: '{input_pdf_path}'")

    for page_num in range(total_pages):
        page = doc[page_num]
        print(f" - [{page_num + 1}/{total_pages}] 페이지 분석 중...")

        # 2. PDF 페이지를 이미지(NumPy 배열)로 변환
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)

        # 3. PaddleOCR 실행
        result = ocr.predict(img_np) #, cls=True)

        # 인식된 텍스트가 없는 빈 페이지 건너뛰기
        if not result or result[0] is None:
            continue

        # 4. 추출된 텍스트와 바운딩 박스 좌표를 PDF 원본 좌표계로 변환하여 투명 삽입
        # =======================================================
        # [수정된 부분] 최신 PaddleX 엔진 및 구버전 데이터 구조 통합 처리
        # =======================================================
        extracted_lines = []

        # 1) 최신 PaddleX 엔진 구조 (Dictionary 형태)인 경우
        if isinstance(result[0], dict) or hasattr(result[0], 'keys'):
            res_dict = dict(result[0])
            # PaddleX 반환 키에서 박스, 텍스트, 신뢰도 리스트 추출
            boxes = res_dict.get('dt_polys', res_dict.get('poly', []))
            texts = res_dict.get('rec_text', res_dict.get('rec_texts', []))
            scores = res_dict.get('rec_score', res_dict.get('rec_scores', []))

            for b, t, s in zip(boxes, texts, scores):
                extracted_lines.append((b, str(t), float(s)))

        # 2) old PaddleOCR 구조 (List 형태)인 경우
        else:
            for item in result[0]:
                box = item[0]
                if isinstance(item[1], (tuple, list)):
                    t = item[1][0]
                    s = item[1][1]
                else:
                    t = item[1]
                    s = item[2] if len(item) > 2 else 1.0
                extracted_lines.append((box, str(t), float(s)))

        # 4. 추출된 텍스트와 바운딩 박스 좌표를 PDF 원본 좌표계로 변환하여 투명 삽입
        for box, text, confidence in extracted_lines:
            # 신뢰도가 너무 낮은 노이즈성 인식 결과 제외
            if confidence < 0.5 or not text.strip():
                continue

            # 이미지 좌표를 PDF 포인트 좌표로 역산
            x0 = min(pt[0] for pt in box) / scale
            y0 = min(pt[1] for pt in box) / scale
            x1 = max(pt[0] for pt in box) / scale
            y1 = max(pt[1] for pt in box) / scale

            box_width = x1 - x0
            box_height = y1 - y0
            if box_width <= 0 or box_height <= 0:
                continue

            # 폰트 크기를 박스 높이에 맞추고, 가로 폭 비율에 맞게 글자 크기 미세 조정
            fontsize = box_height * 0.85
            estimated_width = sum(fontsize * (0.95 if ord(c) > 127 else 0.5) for c in text)

            if estimated_width > box_width:
                fontsize *= (box_width / estimated_width)

            # baseline(글자 하단 기준선) 위치 계산
            insert_point = pymupdf.Point(x0, y1 - (box_height * 0.15))

            # PyMuPDF 내장 한중일(CJK) 폰트를 사용해 투명 텍스트 삽입
            page.insert_text(
                point=insert_point,
                text=text,
                fontsize=max(fontsize, 4),
                fontname="korea",
                render_mode=3
            )

    # 5. 압축 및 최적화 옵션과 함께 새 PDF로 저장
    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()
    print(f"OCR 완료! 검색 가능한 PDF가 저장되었습니다: '{output_pdf_path}'")


if __name__ == "__main__":
    input_file = "C:\\Users\\cas\\Downloads\\Chapter_0_DAP.pdf"
    output_file = "C:\\Users\\cas\\Downloads\\korean_searchable_output.pdf"

    create_searchable_pdf_with_paddle(input_file, output_file, dpi=150)