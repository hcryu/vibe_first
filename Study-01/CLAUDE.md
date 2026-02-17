# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

MNIST 데이터셋으로 학습한 CNN을 이용한 손글씨 숫자 인식 Tkinter GUI 애플리케이션.

## 실행 방법

```bash
# 가상환경 활성화 후 실행
source .venv/bin/activate
python digit_recognizer.py
```

macOS: `digit_recognizer.command` 더블클릭 / Windows: `digit_recognizer.bat` 더블클릭

## 환경

- Python 3.12 (homebrew)
- 가상환경: `.venv/` (프로젝트 로컬)
- 패키지 설치: `pip install -r requirements.txt`

## 핵심 의존성

- **tensorflow** - CNN 모델 학습/추론
- **numpy** - 이미지 배열 처리
- **pillow** - 캔버스 이미지 캡처 및 리사이징
- **tkinter** - GUI (Python 표준 라이브러리)

## 아키텍처

단일 파일(`digit_recognizer.py`) 구조:

- `build_and_train_model()` - MNIST로 CNN 학습 후 `mnist_model.keras`로 저장
- `load_model()` - 저장된 모델 로드, 없으면 자동 학습
- `DigitRecognizerApp` - Tkinter 기반 280x280 캔버스에서 그림 → 28x28로 리사이징 → 모델 추론

모델 파일(`mnist_model.keras`)은 최초 실행 시 자동 생성됨.
