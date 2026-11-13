import os
import fitz  # PyMuPDF
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
    # lang='korean' 설정 시 한국어 + 영어 + 숫자 + 기본 기호를 모두 높은 정확도로 인식합니다.
    # 최초 실행 시 한국어 인식 모델(.tar)을 자동으로 다운로드합니다.
    print("PaddleOCR 한국어 모델 로드 중...")
    ocr = PaddleOCR(
        use_angle_cls=True,  # 90도/180도 회전되거나 기울어진 글자 자동 보정
        lang="korean",       # 한국어 특화 모델 (영어 동시 지원)
        show_log=False       # 불필요한 디버그 로그 숨김
    )

    doc = fitz.open(input_pdf_path)
    total_pages = len(doc)
    scale = dpi / 72.0  # PDF 기본 해상도(72 DPI) 대비 확대 비율
    matrix = fitz.Matrix(scale, scale)

    print(f"총 {total_pages}페이지 OCR 처리 시작: '{input_pdf_path}'")

    for page_num in range(total_pages):
        page = doc[page_num]
        print(f" - [{page_num + 1}/{total_pages}] 페이지 분석 중...")

        # 2. PDF 페이지를 이미지(NumPy 배열)로 변환
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)

        # 3. PaddleOCR 실행
        result = ocr.ocr(img_np, cls=True)

        # 인식된 텍스트가 없는 빈 페이지 건너뛰기
        if not result or result[0] is None:
            continue

        # 4. 추출된 텍스트와 바운딩 박스 좌표를 PDF 원본 좌표계로 변환하여 투명 삽입
        for line in result[0]:
            box = line[0]            # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]        # 인식된 문자열
            confidence = line[1][1]  # 신뢰도 (0.0 ~ 1.0)

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
            # 한글은 1글자당 폭이 폰트 크기와 거의 같고, 영문/공백은 약 절반 폭 차지
            estimated_width = sum(
                fontsize * (0.95 if ord(c) > 127 else 0.5) for c in text
            )
            if estimated_width > box_width:
                fontsize *= (box_width / estimated_width)

            # baseline(글자 하단 기준선) 위치 계산
            insert_point = fitz.Point(x0, y1 - (box_height * 0.15))

            # PyMuPDF 내장 한중일(CJK) 폰트('korea')를 사용해 투명(render_mode=3) 텍스트 삽입
            page.insert_text(
                point=insert_point,
                text=text,
                fontsize=max(fontsize, 4),  # 최소 폰트 크기 보장
                fontname="korea",           # 한글 깨짐 방지용 내장 CJK 폰트 지정
                render_mode=3               # 3 = 화면에는 보이지 않지만 드래그/검색 가능한 투명 텍스트
            )

    # 5. 압축 및 최적화 옵션과 함께 새 PDF로 저장
    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()
    print(f"OCR 완료! 검색 가능한 PDF가 저장되었습니다: '{output_pdf_path}'")


if __name__ == "__main__":
    input_file = "scanned_sample.pdf"
    output_file = "korean_searchable_output.pdf"

    create_searchable_pdf_with_paddle(input_file, output_file, dpi=300)