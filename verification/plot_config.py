"""
Matplotlib 한글 폰트 및 플롯 설정.

모든 검증 케이스에서 import하여 한글 텍스트가 깨지지 않도록 설정.
글로벌 폰트 크기도 여기서 설정 (보고서 캡션과 유사한 크기).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform


def setup_korean_font():
    """한글 폰트를 matplotlib에 설정하고, 글로벌 폰트 크기를 키운다."""

    font_name = None

    if platform.system() == 'Windows':
        # 직접 파일 경로로 먼저 시도 (가장 확실)
        font_path = r'C:\Windows\Fonts\malgun.ttf'
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = 'Malgun Gothic'
        else:
            # 후보 리스트
            for name in ['Malgun Gothic', 'NanumGothic', 'Gulim']:
                try:
                    fp = fm.findfont(fm.FontProperties(family=name))
                    if fp and 'DejaVu' not in fp:
                        font_name = name
                        break
                except Exception:
                    continue
    else:
        for name in ['NanumGothic', 'NanumBarunGothic', 'UnDotum']:
            try:
                fp = fm.findfont(fm.FontProperties(family=name))
                if fp and 'DejaVu' not in fp:
                    font_name = name
                    break
            except Exception:
                continue

    if font_name:
        # font.family + sans-serif 폴백 모두 설정
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans',
                                            'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # ── 글로벌 폰트 크기 설정 (보고서 캡션 ~11pt에 대응) ──
    plt.rcParams.update({
        'font.size': 13,           # 기본 폰트
        'axes.titlesize': 14,      # subplot 제목
        'axes.labelsize': 13,      # 축 라벨
        'xtick.labelsize': 11,     # x축 눈금
        'ytick.labelsize': 11,     # y축 눈금
        'legend.fontsize': 11,     # 범례
        'figure.titlesize': 15,    # suptitle
    })

    return font_name


# 모듈 임포트 시 자동 설정
_FONT_NAME = setup_korean_font()
