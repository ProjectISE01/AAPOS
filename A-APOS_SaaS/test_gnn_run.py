
import os
import simpy
import json
from A_APOS_Engine.data_manager import APOSDataManager
from A_APOS_Engine.engine_wrapper import SimBridge

def run_test_simulation():
    # 1. 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "SMT_2020 - Final", "AutoSched")
    
    print(f"--- GNN Simulation Test Start ---")
    print(f"Data Path: {DATA_PATH}")

    # 2. 데이터 로드 (DS4: LVHM_E 고장 포함 데이터셋)
    dm = APOSDataManager(base_path=DATA_PATH)
    data = dm.load_dataset(4)
    
    # 3. 시뮬레이션 환경 및 브리지 초기화
    env = simpy.Environment()
    bridge = SimBridge(env, data)
    
    print(f"Initialized SimBridge with {len(bridge.stations)} stations.")

    # 4. 시뮬레이션 실행 (500분 진행)
    # 초기에는 Lot 투입 및 대기열 형성이 필요하므로 충분히 돌림
    print("Running simulation for 500 minutes...")
    bridge.run_step(until=500)
    
    # 5. GNN 로그 확인
    ui_state = bridge.update_ui_state()
    gnn_logs = ui_state.get("gnn_logs", [])
    
    print(f"\n--- GNN Action Logs (Last {len(gnn_logs)} events) ---")
    if not gnn_logs:
        print("No GNN logs generated yet. (Waiting for more lots to arrive at stations)")
    else:
        for log in gnn_logs:
            print(f"[GNN] {log}")

    # 6. 설비 상태 요약
    summary = bridge.get_summary()
    print(f"\n--- Simulation Summary at T={int(env.now)} ---")
    print(f"WIP: {summary['wip']} lots")
    print(f"Completed: {summary['completed']} lots")
    print(f"Station States: Busy({summary['busy']}), Down({summary['down']}), Idle({summary['idle']})")

if __name__ == "__main__":
    run_test_simulation()
