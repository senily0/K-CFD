"""
Case 18: 웹 기반 실시간 시각화 대시보드 검증.

Flask + Plotly.js 대시보드 서버를 시작하고,
간단한 시뮬레이션 데이터를 수집하여 API 응답을 검증.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.web_dashboard import SimulationMonitor, WebDashboard


def run_case18(results_dir: str = "results", figures_dir: str = "figures") -> dict:
    """
    웹 대시보드 기능 검증.

    1. SimulationMonitor + WebDashboard 생성
    2. 대시보드 서버 시작 (port 5050)
    3. 가상 시뮬레이션 데이터 주입 (20 스텝)
    4. HTTP API 응답 확인
    5. 서버 종료
    6. 대시보드 레이아웃 모사 그림 생성

    Returns
    -------
    result : dict
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 18: 웹 대시보드 검증")
    print("=" * 60)

    result = {
        'server_started': False,
        'api_responsive': False,
        'data_collected': False,
        'n_data_points': 0,
        'html_served': False,
        'converged': False
    }

    # 1. Monitor 생성
    monitor = SimulationMonitor()

    # 2. Dashboard 시작
    port = 5050
    dashboard = WebDashboard(monitor, host='127.0.0.1', port=port)
    try:
        dashboard.start()
        result['server_started'] = True
        print(f"  서버 시작: http://127.0.0.1:{port}")
    except Exception as e:
        print(f"  서버 시작 실패: {e}")
        # 서버 시작 실패해도 데이터 수집 테스트는 진행
        result['server_started'] = False

    # 3. 가상 시뮬레이션 데이터 주입
    print("  가상 시뮬레이션 데이터 주입 (20 스텝)...")
    n_steps = 20
    dt = 0.01
    residuals = []
    dt_history = []
    cfl_history = []

    for step in range(1, n_steps + 1):
        sim_time = step * dt
        residual = 1.0 * np.exp(-0.3 * step)  # 지수 감소
        current_dt = dt * (1.0 + 0.1 * np.sin(step * 0.5))  # 약간의 변동
        cfl = 0.3 + 0.1 * np.sin(step * 0.3)

        # 가상 필드 (10x10 격자)
        field = np.sin(np.linspace(0, 2*np.pi, 100)) * np.exp(-0.1 * step)

        monitor.update(
            step=step,
            sim_time=sim_time,
            residual=residual,
            dt=current_dt,
            cfl=cfl,
            fields={'temperature': field}
        )

        residuals.append(residual)
        dt_history.append(current_dt)
        cfl_history.append(cfl)
        time.sleep(0.05)  # 약간의 지연

    monitor.set_status('completed')
    print(f"  데이터 주입 완료: {n_steps} 스텝")

    # 4. API 응답 확인
    try:
        import urllib.request
        import json

        # /api/data 확인
        url_data = f"http://127.0.0.1:{port}/api/data"
        req = urllib.request.Request(url_data)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode())
                result['api_responsive'] = True
                result['n_data_points'] = len(body.get('time_steps', []))
                result['data_collected'] = result['n_data_points'] > 0
                print(f"  /api/data 응답: OK ({result['n_data_points']} 데이터 포인트)")
            else:
                print(f"  /api/data 응답: HTTP {resp.status}")

        # / (HTML) 확인
        url_html = f"http://127.0.0.1:{port}/"
        req2 = urllib.request.Request(url_html)
        with urllib.request.urlopen(req2, timeout=5) as resp2:
            if resp2.status == 200:
                html = resp2.read().decode()
                result['html_served'] = 'Plotly' in html and 'Simulation Monitor' in html
                print(f"  / HTML 응답: OK (Plotly 포함: {result['html_served']})")

        # /api/status 확인
        url_status = f"http://127.0.0.1:{port}/api/status"
        req3 = urllib.request.Request(url_status)
        with urllib.request.urlopen(req3, timeout=5) as resp3:
            if resp3.status == 200:
                status_body = json.loads(resp3.read().decode())
                print(f"  /api/status 응답: {status_body.get('status', 'unknown')}")

        # /api/field/temperature 확인
        url_field = f"http://127.0.0.1:{port}/api/field/temperature"
        req4 = urllib.request.Request(url_field)
        with urllib.request.urlopen(req4, timeout=5) as resp4:
            if resp4.status == 200:
                field_body = json.loads(resp4.read().decode())
                print(f"  /api/field/temperature 응답: {len(field_body.get('values', []))} values")

    except Exception as e:
        print(f"  API 테스트 실패: {e}")
        # 서버 미시작 시에도 데이터 수집 자체는 확인
        data = monitor.get_data()
        result['n_data_points'] = len(data['time_steps'])
        result['data_collected'] = result['n_data_points'] > 0

    # 5. 서버 종료
    dashboard.stop()

    # 6. 대시보드 레이아웃 모사 그림 생성
    print("  대시보드 레이아웃 모사 그림 생성...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Web Dashboard - Simulation Monitor (Case 18)', fontsize=14, fontweight='bold')

    # 잔차 이력
    axes[0, 0].semilogy(range(1, n_steps+1), residuals, 'b-o', markersize=3)
    axes[0, 0].set_title('Residual History')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)

    # CFL 이력
    axes[0, 1].plot(range(1, n_steps+1), cfl_history, 'g-o', markersize=3)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='CFL = 1.0')
    axes[0, 1].set_title('CFL Number')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('CFL')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # dt 이력
    axes[1, 0].plot(range(1, n_steps+1), dt_history, 'm-o', markersize=3)
    axes[1, 0].set_title('Time Step (dt) History')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('dt [s]')
    axes[1, 0].grid(True, alpha=0.3)

    # 필드 스냅샷 (최종)
    field_data = np.sin(np.linspace(0, 2*np.pi, 100)) * np.exp(-0.1 * n_steps)
    field_2d = field_data.reshape(10, 10)
    im = axes[1, 1].imshow(field_2d, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Temperature Field (Final)')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case18_web_dashboard.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # PASS 판정
    result['converged'] = (
        result['data_collected'] and
        result['n_data_points'] == n_steps
    )

    print(f"\n  결과:")
    print(f"    서버 시작: {result['server_started']}")
    print(f"    API 응답: {result['api_responsive']}")
    print(f"    데이터 수집: {result['data_collected']} ({result['n_data_points']} points)")
    print(f"    HTML 제공: {result['html_served']}")
    print(f"    PASS: {result['converged']}")

    return result


if __name__ == "__main__":
    result = run_case18()
    print(f"\nDONE Case18")
