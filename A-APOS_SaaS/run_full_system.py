
import os
import simpy
import json
import pandas as pd
from A_APOS_Engine.data_manager import APOSDataManager
from A_APOS_Engine.engine_wrapper import SimBridge

def run_integrated_simulation():
    # 1. 경로 및 환경 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "SMT_2020 - Final", "AutoSched")
    
    print("="*60)
    print("🚀 A-APOS Integrated AI Engine Simulation")
    print("="*60)
    print(f"Data Path: {DATA_PATH}")

    # 2. 데이터 로드 (DS4: LVHM_E - 고장/복합 공정 포함)
    dm = APOSDataManager(base_path=DATA_PATH)
    data = dm.load_dataset(4)
    
    # 3. 시뮬레이션 환경 및 브리지 초기화
    env = simpy.Environment()
    bridge = SimBridge(env, data)
    
    print(f"[*] Initialized SimBridge with {len(bridge.stations)} stations.")
    print(f"[*] GNN Dispatcher: Enabled (CQT Urgency EWS)")
    print(f"[*] XGBoost Predictor: Enabled (83-Feature Pipeline)")
    print(f"[*] Flow Control: Enabled (Conditional WIP Floodgate)")
    print("-"*60)

    # 4. 시뮬레이션 루프 실행 (1000분 단위로 진행하며 상태 모니터링)
    total_time = 10000 # 총 10000분 시뮬레이션으로 대폭 확대
    step_size = 1000
    
    for current_until in range(step_size, total_time + 1, step_size):
        print(f"\n▶ Simulating until T={current_until}...")
        ui_state = bridge.run_step(until=current_until)
        
        # KPI 출력
        kpi = ui_state['kpi']
        print(f"   [Status] WIP: {ui_state['wip']} | Completed: {kpi['completed']} | Down: {kpi['down_count']}")
        
        # GNN/CQT/XGB 로그 중 중요한 것 필터링해서 출력
        gnn_logs = ui_state.get("gnn_logs", [])
        # 모든 AI 로그 출력 (검증용)
        relevant_logs = [log for log in gnn_logs if any(x in log for x in ["CRITICAL", "URGENT", "XGB 83F", "Warning"])]
        if relevant_logs:
            print(f"   [AI Actions] Recent events (Total: {len(relevant_logs)}):")
            for log in relevant_logs[-5:]:
                print(f"     - {log}")

    # 5. 최종 요약
    print("\n" + "="*60)
    print("✅ Simulation Completed Successfully")
    summary = bridge.get_summary()
    print(f"Final WIP: {summary['wip']}")
    print(f"Total Completed: {summary['completed']}")
    print(f"Final Cycle Time (Avg): {ui_state['kpi']['avg_ct']} hours")
    print(f"On-time Delivery Rate: {ui_state['kpi']['ontime_pct']}%")
    print("="*60)

if __name__ == "__main__":
    run_integrated_simulation()
