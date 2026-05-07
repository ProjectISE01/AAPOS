
import os
import simpy
import json
import numpy as np
from A_APOS_Engine.data_manager import APOSDataManager
from A_APOS_Engine.engine_wrapper import SimBridge

def run_test_simulation():
    # 1. 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "SMT_2020 - Final", "AutoSched")
    
    print(f"--- Hybrid AI (GNN + XGBoost) Simulation Test Start ---")
    print(f"Data Path: {DATA_PATH}")

    # 2. 데이터 로드 (DS4: LVHM_E 고장 포함 데이터셋)
    dm = APOSDataManager(base_path=DATA_PATH)
    data = dm.load_dataset(4)
    
    # 3. 시뮬레이션 환경 및 브리지 초기화
    env = simpy.Environment()
    bridge = SimBridge(env, data)
    
    print(f"Initialized SimBridge with {len(bridge.stations)} stations.")

    # 4. 시뮬레이션 실행 (여러 스텝으로 나누어 실행하여 XGBoost 히스토리 축적)
    print("Running simulation steps to build XGBoost history...")
    for i in range(15): # 기간을 조금 늘림
        bridge.run_step(until=(i+1)*200)
        ui_state = bridge.update_ui_state()
        gnn_logs = ui_state.get("gnn_logs", [])
        xgb_probs = ui_state.get("xgb_probs", {})
        
        # 모든 AI 로그 출력
        if gnn_logs:
            print(f"[T={(i+1)*200}] AI Logs: {gnn_logs}")
        
        # 특정 구역(예: Litho)의 확률 출력
        if "Litho" in xgb_probs:
            print(f"[T={(i+1)*200}] Litho Bottleneck Prob: {xgb_probs['Litho']*100:.1f}%")

    # 5. 최종 결과 확인
    ui_state = bridge.update_ui_state()
    gnn_logs = ui_state.get("gnn_logs", [])
    
    print(f"\n--- Recent AI Console Logs ---")
    for log in gnn_logs[-10:]:
        print(f"[AI] {log}")

    # 6. 설비 상태 요약
    summary = bridge.get_summary()
    print(f"\n--- Simulation Summary at T={int(env.now)} ---")
    print(f"WIP: {summary['wip']} lots")
    print(f"Completed: {summary['completed']} lots")
    print(f"Station States: Busy({summary['busy']}), Down({summary['down']}), Idle({summary['idle']})")

if __name__ == "__main__":
    run_test_simulation()
