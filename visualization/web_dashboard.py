"""
웹 기반 실시간 시뮬레이션 모니터링 대시보드.

Flask + Plotly.js로 시뮬레이션 진행 상황을 실시간 표시.
- 잔차 수렴 이력
- CFL 수 이력 (적응 시간 간격 사용 시)
- dt 변화 이력
- 필드 2D 컬러맵 (최신 스냅샷)
"""

import threading
import time
import json
import numpy as np

from flask import Flask, render_template_string, jsonify


class SimulationMonitor:
    """
    시뮬레이션 데이터 수집기 (thread-safe).

    시뮬레이션 루프에서 매 스텝 update()를 호출하면
    대시보드가 get_data()로 polling하여 표시.
    """

    def __init__(self):
        self.data = {
            'time_steps': [],
            'times': [],
            'residuals': [],
            'dt_history': [],
            'cfl_history': [],
            'field_snapshots': {},
            'status': 'idle',
            'current_step': 0,
            'current_time': 0.0,
            'start_wall_time': None,
            'elapsed_wall_time': 0.0
        }
        self._lock = threading.Lock()

    def update(self, step, sim_time, residual, dt=None, cfl=None, fields=None):
        """
        시뮬레이션에서 매 스텝 호출.

        Parameters
        ----------
        step : int
        sim_time : float
        residual : float
        dt : float, optional
        cfl : float, optional
        fields : dict of {name: np.ndarray}, optional
        """
        with self._lock:
            if self.data['start_wall_time'] is None:
                self.data['start_wall_time'] = time.time()
            self.data['status'] = 'running'
            self.data['current_step'] = step
            self.data['current_time'] = float(sim_time)
            self.data['elapsed_wall_time'] = time.time() - self.data['start_wall_time']
            self.data['time_steps'].append(step)
            self.data['times'].append(float(sim_time))
            self.data['residuals'].append(float(residual))
            if dt is not None:
                self.data['dt_history'].append(float(dt))
            if cfl is not None:
                self.data['cfl_history'].append(float(cfl))
            if fields is not None:
                for name, arr in fields.items():
                    self.data['field_snapshots'][name] = arr.tolist() if isinstance(arr, np.ndarray) else arr

    def set_status(self, status):
        """상태 변경: 'idle', 'running', 'completed', 'error'."""
        with self._lock:
            self.data['status'] = status
            if status == 'completed':
                self.data['elapsed_wall_time'] = (
                    time.time() - self.data['start_wall_time']
                    if self.data['start_wall_time'] else 0.0
                )

    def get_data(self):
        """대시보드에서 polling. thread-safe copy 반환."""
        with self._lock:
            return {
                'time_steps': list(self.data['time_steps']),
                'times': list(self.data['times']),
                'residuals': list(self.data['residuals']),
                'dt_history': list(self.data['dt_history']),
                'cfl_history': list(self.data['cfl_history']),
                'status': self.data['status'],
                'current_step': self.data['current_step'],
                'current_time': self.data['current_time'],
                'elapsed_wall_time': self.data['elapsed_wall_time'],
                'field_names': list(self.data['field_snapshots'].keys())
            }

    def get_field(self, name):
        """특정 필드 스냅샷 반환."""
        with self._lock:
            return self.data['field_snapshots'].get(name, None)


# HTML 대시보드 템플릿 (Plotly.js CDN 사용)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FVM Simulation Monitor</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0; padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: #2c3e50; color: white;
            padding: 15px 25px; margin: -20px -20px 20px -20px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .header h1 { margin: 0; font-size: 1.4em; }
        .status-badge {
            padding: 5px 15px; border-radius: 20px;
            font-weight: bold; font-size: 0.9em;
        }
        .status-idle { background: #95a5a6; }
        .status-running { background: #27ae60; }
        .status-completed { background: #2980b9; }
        .status-error { background: #e74c3c; }
        .info-bar {
            display: flex; gap: 30px; margin-bottom: 20px;
            background: white; padding: 15px 20px;
            border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .info-item { text-align: center; }
        .info-item .label { font-size: 0.8em; color: #7f8c8d; }
        .info-item .value { font-size: 1.3em; font-weight: bold; color: #2c3e50; }
        .charts {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .chart-box {
            background: white; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 10px;
        }
        @media (max-width: 900px) {
            .charts { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Two-Fluid FVM Simulation Monitor</h1>
        <span id="statusBadge" class="status-badge status-idle">IDLE</span>
    </div>

    <div class="info-bar">
        <div class="info-item">
            <div class="label">Time Step</div>
            <div class="value" id="stepVal">0</div>
        </div>
        <div class="info-item">
            <div class="label">Simulation Time</div>
            <div class="value" id="timeVal">0.000 s</div>
        </div>
        <div class="info-item">
            <div class="label">Wall Time</div>
            <div class="value" id="wallVal">0.0 s</div>
        </div>
        <div class="info-item">
            <div class="label">Latest Residual</div>
            <div class="value" id="resVal">-</div>
        </div>
        <div class="info-item">
            <div class="label">Current dt</div>
            <div class="value" id="dtVal">-</div>
        </div>
    </div>

    <div class="charts">
        <div class="chart-box"><div id="residualChart"></div></div>
        <div class="chart-box"><div id="cflChart"></div></div>
        <div class="chart-box"><div id="dtChart"></div></div>
        <div class="chart-box"><div id="fieldChart"></div></div>
    </div>

    <script>
        // Initialize empty charts
        Plotly.newPlot('residualChart', [{x:[], y:[], type:'scatter', name:'Residual'}], {
            title: 'Residual History', yaxis: {type:'log', title:'Residual'},
            xaxis: {title:'Step'}, margin: {t:40, b:40, l:60, r:20}
        }, {responsive: true});

        Plotly.newPlot('cflChart', [
            {x:[], y:[], type:'scatter', name:'CFL'},
            {x:[], y:[], type:'scatter', name:'CFL=1.0', line:{dash:'dash', color:'red'}}
        ], {
            title: 'CFL Number', yaxis: {title:'CFL'},
            xaxis: {title:'Step'}, margin: {t:40, b:40, l:60, r:20}
        }, {responsive: true});

        Plotly.newPlot('dtChart', [{x:[], y:[], type:'scatter', name:'dt'}], {
            title: 'Time Step History', yaxis: {type:'log', title:'dt [s]'},
            xaxis: {title:'Step'}, margin: {t:40, b:40, l:60, r:20}
        }, {responsive: true});

        Plotly.newPlot('fieldChart', [{z:[[0]], type:'heatmap', colorscale:'Viridis'}], {
            title: 'Field Snapshot', margin: {t:40, b:40, l:60, r:20}
        }, {responsive: true});

        function updateDashboard() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    // Status badge
                    const badge = document.getElementById('statusBadge');
                    badge.textContent = data.status.toUpperCase();
                    badge.className = 'status-badge status-' + data.status;

                    // Info values
                    document.getElementById('stepVal').textContent = data.current_step;
                    document.getElementById('timeVal').textContent = data.current_time.toFixed(4) + ' s';
                    document.getElementById('wallVal').textContent = data.elapsed_wall_time.toFixed(1) + ' s';

                    if (data.residuals.length > 0)
                        document.getElementById('resVal').textContent = data.residuals[data.residuals.length-1].toExponential(2);
                    if (data.dt_history.length > 0)
                        document.getElementById('dtVal').textContent = data.dt_history[data.dt_history.length-1].toExponential(3);

                    // Residual chart
                    const steps = data.time_steps;
                    Plotly.react('residualChart', [{x:steps, y:data.residuals, type:'scatter'}], {
                        title:'Residual History', yaxis:{type:'log', title:'Residual'},
                        xaxis:{title:'Step'}, margin:{t:40,b:40,l:60,r:20}
                    });

                    // CFL chart
                    if (data.cfl_history.length > 0) {
                        const cflSteps = steps.slice(0, data.cfl_history.length);
                        const cflLimit = cflSteps.map(() => 1.0);
                        Plotly.react('cflChart', [
                            {x:cflSteps, y:data.cfl_history, type:'scatter', name:'CFL'},
                            {x:cflSteps, y:cflLimit, type:'scatter', name:'Limit', line:{dash:'dash',color:'red'}}
                        ], {title:'CFL Number', yaxis:{title:'CFL'}, xaxis:{title:'Step'}, margin:{t:40,b:40,l:60,r:20}});
                    }

                    // dt chart
                    if (data.dt_history.length > 0) {
                        const dtSteps = steps.slice(0, data.dt_history.length);
                        Plotly.react('dtChart', [{x:dtSteps, y:data.dt_history, type:'scatter'}], {
                            title:'Time Step History', yaxis:{type:'log', title:'dt [s]'},
                            xaxis:{title:'Step'}, margin:{t:40,b:40,l:60,r:20}
                        });
                    }

                    // Field heatmap
                    if (data.field_names.length > 0) {
                        fetch('/api/field/' + data.field_names[0])
                            .then(r => r.json())
                            .then(fdata => {
                                if (fdata.values) {
                                    // Reshape 1D to 2D (approximate)
                                    const n = fdata.values.length;
                                    const cols = Math.ceil(Math.sqrt(n));
                                    const rows = Math.ceil(n / cols);
                                    let z = [];
                                    for (let r = 0; r < rows; r++) {
                                        let row = [];
                                        for (let c = 0; c < cols; c++) {
                                            let idx = r * cols + c;
                                            row.push(idx < n ? fdata.values[idx] : 0);
                                        }
                                        z.push(row);
                                    }
                                    Plotly.react('fieldChart',
                                        [{z:z, type:'heatmap', colorscale:'Viridis'}],
                                        {title:'Field: ' + data.field_names[0], margin:{t:40,b:40,l:60,r:20}}
                                    );
                                }
                            });
                    }
                })
                .catch(e => console.log('Update failed:', e));
        }

        // Poll every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""


class WebDashboard:
    """
    Flask 웹 대시보드 서버.

    SimulationMonitor에서 데이터를 읽어 Plotly.js 대시보드로 표시.
    백그라운드 스레드에서 실행.

    Usage
    -----
    >>> monitor = SimulationMonitor()
    >>> dashboard = WebDashboard(monitor, port=5000)
    >>> dashboard.start()       # 백그라운드 서버 시작
    >>> # ... 시뮬레이션 실행 중 monitor.update() 호출 ...
    >>> dashboard.stop()        # 서버 종료
    """

    def __init__(self, monitor, host='127.0.0.1', port=5000):
        self.monitor = monitor
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._thread = None
        self._setup_routes()

    def _setup_routes(self):
        """Flask 라우트 설정."""
        monitor = self.monitor

        @self.app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)

        @self.app.route('/api/data')
        def api_data():
            return jsonify(monitor.get_data())

        @self.app.route('/api/status')
        def api_status():
            data = monitor.get_data()
            return jsonify({
                'status': data['status'],
                'current_step': data['current_step'],
                'current_time': data['current_time'],
                'elapsed_wall_time': data['elapsed_wall_time']
            })

        @self.app.route('/api/field/<name>')
        def api_field(name):
            values = monitor.get_field(name)
            if values is None:
                return jsonify({'error': f'Field {name} not found'}), 404
            return jsonify({'name': name, 'values': values})

    def start(self):
        """백그라운드 스레드로 서버 시작."""
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self._thread = threading.Thread(
            target=lambda: self.app.run(
                host=self.host, port=self.port,
                debug=False, use_reloader=False
            ),
            daemon=True
        )
        self._thread.start()
        time.sleep(0.5)  # 서버 시작 대기
        print(f"  [Dashboard] http://{self.host}:{self.port} 에서 대시보드 실행 중")

    def stop(self):
        """서버 종료 (daemon thread이므로 프로세스 종료 시 자동 종료)."""
        print("  [Dashboard] 서버 종료")
        # Flask dev server doesn't have clean shutdown in threads
        # The daemon thread will be killed when the main process exits

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"
