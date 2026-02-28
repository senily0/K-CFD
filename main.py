"""
Two-Fluid Model FVM 기반 열유체 코드 - 메인 실행 스크립트.

전체 검증 케이스 실행 및 DOCX 보고서 생성.
"""

import os
import sys
import time

# 작업 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from verification.run_all import run_all_cases
from report.report_generator import generate_report
from visualization.vtu_renderer import render_all_missing


def main():
    """메인 실행."""
    print("=" * 70)
    print("  Two-Fluid Model FVM 열유체 코드")
    print("  Euler-Euler 이상유동 + CHT + k-epsilon 난류")
    print("=" * 70)

    results_dir = os.path.join(BASE_DIR, "results")
    figures_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    total_start = time.time()

    # 1) 전체 검증 실행
    print("\n[1/2] 검증 케이스 실행")
    all_results = run_all_cases(results_dir, figures_dir)

    # 2) ParaView 스타일 렌더링 생성
    print("\n[2/3] ParaView 스타일 렌더링 생성")
    try:
        render_all_missing(results_dir, figures_dir)
    except Exception as e:
        print(f"  렌더링 생성 실패: {e}")

    # 3) 보고서 생성
    print("\n[3/3] DOCX 보고서 생성")
    report_dir = os.path.join(BASE_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "TwoFluid_FVM_Report.docx")
    try:
        generate_report(all_results, report_path, figures_dir)
        print(f"  보고서 생성 완료: {report_path}")
    except Exception as e:
        print(f"  보고서 생성 실패: {e}")

    total_elapsed = time.time() - total_start
    print(f"\n  전체 소요 시간: {total_elapsed:.1f}초")
    print("=" * 70)
    print("  완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
