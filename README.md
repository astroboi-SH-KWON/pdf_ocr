# pdf_ocr


# 1. 가상환경 생성 (PaddleOCR에 python=3.10이 가장 안정적)
conda create -n paddle_ocr_env python=3.10 -y

# 2. 가상환경 활성화
conda activate paddle_ocr_env

# 3. CPU 버전 PaddlePaddle 딥러닝 프레임워크 설치
pip install paddlepaddle

# 4. PaddleOCR 및 PDF 처리용 PyMuPDF, 이미지 처리용 Pillow/NumPy 설치
pip install "paddleocr>=2.7.0" pymupdf pillow numpy
