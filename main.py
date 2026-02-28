"""Two-Fluid FVM — 대안 실행 진입점.

전체 검증 케이스 실행 및 보고서 생성은 generate_report.py를 참조.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_report import main

if __name__ == "__main__":
    main()
