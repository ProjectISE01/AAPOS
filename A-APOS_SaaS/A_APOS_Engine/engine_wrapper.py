"""
engine_wrapper.py — SimPy ↔ Streamlit 브리지 (Final)

실제 데이터 구조 (직접 확인):
route_*.txt 컬럼:
  ROUTE(0), STEP(1), DESC(2), STNFAM(3), PDIST(4), PTIME(5), PTIME2(6),
  PTUNITS(7), PTPER(8), BATCHMN(9), BATCHMX(10), SETUP(11), WHEN(12),
  STIME(13), STUNITS(14), ..., IGNORE(28)

Excel Route 컬럼 (검증):
  ROUTE(0), STEP(1), DESC(2), AREA(3), TOOLGROUP(4), PROCESSING UNIT(5),
  DIST(6), MEAN(7), OFFSET(8), UNITS(9), ..., BATCH MIN(12), BATCH MAX(13),
  SETUP(14), WHEN(15), SETUP DIST(16), SETUP TIME(17)

→ txt: PTPER = per_batch/per_piece/per_lot
   Excel: PROCESSING UNIT = Batch/Wafer/Lot
   동일한 개념, 매핑 적용

DS별 product/route 매핑:
  DS1(HVLM):   Product_3, Product_4 → route_3, route_4
  DS2(LVHM):   Product_1~10         → route_1~10
  DS3(HVLM_E): Product_3, Product_4 + Engineering → route_3, route_4, route_E3
  DS4(LVHM_E): Product_1~10 + Engineering          → route_1~10, route_E1~E3
"""
import simpy
import random
import numpy as np
import pandas as pd
from datetime import datetime
from .factory_engine import AdvancedStation, Lot, failure_process
from .gnn_dispatcher import GNNDispatcher
from .xgb_predictor import XGBBottleneckPredictor

BASE_DATE = datetime(2018, 1, 1)

# ── 구역별 고장 파라미터 (Excel Breakdown 시트 기반) ─────────────────────────
BREAKDOWN_TABLE = {
    "Def_Met":    (10080, 35.28),
    "Dielectric": (10080, 604.8),
    "Diffusion":  (10080, 151.2),
    "Dry_Etch":   (10080, 231.84),
    "Implant":    (10080, 604.8),
    "Litho":      (10080, 705.59),
    "Litho_Met":  (10080, 35.28),
    "Planar":     (10080, 201.6),
    "TF":         (10080, 453.6),
    "TF_Met":     (10080, 35.28),
    "Wet_Etch":   (10080, 221.76),
}

# PTPER txt값 → PROCESSING UNIT 매핑
PTPER_TO_UNIT = {
    "per_batch": "Batch",
    "per_piece": "Wafer",
    "per_lot":   "Lot",
    "Batch":     "Batch",
    "Wafer":     "Wafer",
    "Lot":       "Lot",
}


def _sf(val, default=0.0) -> float:
    try:
        s = str(val).strip()
        return default if s in ('', 'nan', 'NaN', 'None') else float(s)
    except (TypeError, ValueError):
        return default


def _ss(val) -> str:
    s = str(val).strip() if val is not None else ''
    return '' if s in ('nan', 'NaN', 'None') else s


def _get_area(stn_name: str, area_hint: str = "") -> str:
    """IGNORE 컬럼(구역명) 또는 설비명으로 구역 판별"""
    h = area_hint.strip()
    if h and h in BREAKDOWN_TABLE:
        return h
    n = stn_name.lower()
    for key, area in [
        ("diffusion",  "Diffusion"),
        ("de_",        "Dry_Etch"), ("dry_etch", "Dry_Etch"),
        ("lithotrack", "Litho"), ("litho_met", "Litho_Met"),
        ("litho_reg",  "Litho_Met"), ("lithomet", "Litho_Met"),
        ("litho_be",   "Litho"), ("litho",   "Litho"),
        ("implant",    "Implant"), ("epi",    "Implant"),
        ("dielectric", "Dielectric"),
        ("planar",     "Planar"), ("cmp",    "Planar"),
        ("tf_met",     "TF_Met"), ("tfmet",  "TF_Met"),
        ("tf_",        "TF"), ("tf",         "TF"),
        ("we_",        "Wet_Etch"), ("wet_etch", "Wet_Etch"),
        ("defmet",     "Def_Met"), ("def_met", "Def_Met"),
    ]:
        if key in n:
            return area
    return "Dry_Etch"


def _parse_route_df(df: pd.DataFrame) -> list:
    """
    route_*.txt DataFrame → 스텝 리스트
    컬럼: STNFAM, PTIME, PTPER, BATCHMN, BATCHMX, SETUP, STIME, IGNORE
    """
    steps = []
    col = {c.upper().strip(): c for c in df.columns}

    stn_col   = col.get('STNFAM')
    ptime_col = col.get('PTIME')
    ptper_col = col.get('PTPER')
    bmin_col  = col.get('BATCHMN')
    bmax_col  = col.get('BATCHMX')
    setup_col = col.get('SETUP')
    stime_col = col.get('STIME')
    ignore_col= col.get('IGNORE')

    if not stn_col or not ptime_col:
        return steps

    for _, row in df.iterrows():
        stn = _ss(row.get(stn_col, ''))
        if not stn:
            continue

        ptime      = max(0.01, _sf(row.get(ptime_col), 1.0))
        ptper_raw  = _ss(row.get(ptper_col, 'per_lot')) if ptper_col else 'per_lot'
        proc_unit  = PTPER_TO_UNIT.get(ptper_raw, 'Lot')
        setup_req  = _ss(row.get(setup_col, '')) if setup_col else ''
        setup_cost = _sf(row.get(stime_col), 0.0) if stime_col else 0.0
        bmin       = _sf(row.get(bmin_col), 0.0) if bmin_col else 0.0
        bmax       = _sf(row.get(bmax_col), bmin) if bmax_col else bmin
        area_hint  = _ss(row.get(ignore_col, '')) if ignore_col else ''

        steps.append({
            "station":         stn,
            "ptime":           ptime,
            "proc_unit":       proc_unit,
            "setup":           setup_req,
            "setup_cost":      setup_cost,
            "is_batch":        proc_unit == "Batch",
            "batch_min_wafers": int(bmin) if bmin > 0 else 1,
            "batch_max_wafers": int(bmax) if bmax > 0 else 1,
            "area":            _get_area(stn, area_hint),
        })
    return steps


class SimBridge:
    def __init__(self, env: simpy.Environment, data: dict, overrides: dict = None):
        self.env            = env
        self.data           = data
        self.overrides      = overrides or {}
        self.stations: dict[str, AdvancedStation] = {}
        self.active_lots:    list = []
        self.completed_lots: list = []
        
        # 디스패칭 정책 설정 (기본값: GNN)
        policy = self.overrides.get("policy", "GNN")
        self.gnn_dispatcher = GNNDispatcher(policy=policy) 
        
        self.xgb_predictor  = XGBBottleneckPredictor(threshold=0.7) # XGBoost 예측기 초기화
        self.kpi_tracker = {
            "completed":    0,
            "cycle_times":  [],
            "ontime_count": 0,
        }
        self.wip_history: list = []
        self.kpi_history: list = []
        self._stn_area:   dict[str, str] = {}
        self.release_events: list = []  # 로트 투입 시점 기록 (최근 투입률 계산용)
        self.gate_control_active = False # 수문 제어 상태 추적
        self.gate_logs = [] # 수문 제어 관련 로그 버퍼

        # ── 1. Route 파싱 ────────────────────────────────────────────
        self.route_steps: dict[str, list] = {}
        for key, df in data["routes"].items():
            steps = _parse_route_df(df)
            if steps:
                self.route_steps[key] = steps
                for s in steps:
                    self._stn_area[s["station"]] = s["area"]

        # ── 2. 설비 초기화 ───────────────────────────────────────────
        stn_cfg: dict[str, dict] = {}
        for steps in self.route_steps.values():
            for s in steps:
                name = s["station"]
                if name not in stn_cfg:
                    stn_cfg[name] = {
                        "is_batch":         False,
                        "batch_min_wafers": 1,
                        "batch_max_wafers": 1,
                        "capacity":         1,
                    }
                if s["is_batch"]:
                    stn_cfg[name]["is_batch"] = True
                    # 가장 작은 min 값 사용
                    cur = stn_cfg[name]["batch_min_wafers"]
                    if cur == 1 or s["batch_min_wafers"] < cur:
                        stn_cfg[name]["batch_min_wafers"] = s["batch_min_wafers"]
                        stn_cfg[name]["batch_max_wafers"] = s["batch_max_wafers"]

        # tool_capacity: {toolgroup: 설비 수} — data_manager에서 tool.txt 파싱
        tool_capacity = data.get("tool_capacity", {})
        cap_factor = self.overrides.get("capacity_factor", 1.0)

        # ── 2-1. Station별 Mean PTime 계산 ──────────────────────────
        stn_ptimes = {}
        for steps in self.route_steps.values():
            for s in steps:
                name = s["station"]
                if name not in stn_ptimes:
                    stn_ptimes[name] = []
                stn_ptimes[name].append(s["ptime"])

        for name, cfg in stn_cfg.items():
            # 실제 설비 대수를 capacity로 반영 (없으면 기본값 1)
            raw_cap = tool_capacity.get(name, 1)
            capacity = max(1, int(raw_cap * cap_factor))
            
            self.stations[name] = AdvancedStation(
                env, name,
                capacity=capacity,
                is_batch=cfg["is_batch"],
                batch_min_wafers=cfg["batch_min_wafers"],
                batch_max_wafers=cfg["batch_max_wafers"],
                dispatcher=self.gnn_dispatcher # 디스패처 전달
            )
            # 평균 처리 시간 설정
            if name in stn_ptimes and stn_ptimes[name]:
                self.stations[name].mean_ptime = np.mean(stn_ptimes[name])

        # ── 3. 고장 프로세스 (DS3, DS4) ─────────────────────────────
        if data.get("downs") is not None:
            mttf_factor = self.overrides.get("mttf_factor", 1.0)
            mttr_factor = self.overrides.get("mttr_factor", 1.0)
            
            area_bd = self._parse_downcal(data["downs"])
            for stn_name, stn_obj in self.stations.items():
                area = self._stn_area.get(stn_name, "Dry_Etch")
                mttf, mttr = area_bd.get(area, BREAKDOWN_TABLE.get(area, (10080, 200)))
                
                # Overrides 적용
                mttf *= mttf_factor
                mttr *= mttr_factor
                
                env.process(failure_process(env, stn_obj, mttf, mttr))

        # ── 4. Lot 투입 등록 ─────────────────────────────────────────
        if self.route_steps:
            env.process(self._release_controller())
            # CQT 모니터링 프로세스 (매 1분마다 전역 상태 업데이트 시뮬레이션)
            env.process(self._cqt_monitor())

    def _cqt_monitor(self):
        """매 분마다 CQT 상태를 체크하고 UI 로그에 경고 등을 남김 (EWS 개념)"""
        while True:
            yield self.env.timeout(1.0)
            # 사실 GNN에서 실시간 계산하므로 여기선 모니터링용 로그만 처리
            urgent_count = 0
            for lot in self.active_lots:
                if lot.cqt_deadline and lot.cqt_deadline - self.env.now < 30:
                    urgent_count += 1
            
            # 10분마다 브로드캐스트 로그 (예시)
            if int(self.env.now) % 100 == 0 and urgent_count > 0:
                pass # UI 로그 등에 추가 가능

    def _parse_downcal(self, downs_df) -> dict:
        result = {}
        if downs_df is None or downs_df.empty:
            return result
        col = {c.upper().strip(): c for c in downs_df.columns}
        ignore_col = col.get('IGNORE')
        mttf_col   = col.get('MTTF')
        mttr_col   = col.get('MTTR')
        if not (ignore_col and mttf_col and mttr_col):
            return result
        for _, row in downs_df.iterrows():
            area = _ss(row.get(ignore_col, ''))
            mttf = _sf(row.get(mttf_col), 10080)
            mttr = _sf(row.get(mttr_col), 200)
            if area:
                result[area] = (mttf, mttr)
        return result

    # ── Lot 공정 흐름 ────────────────────────────────────────────────
    def _lot_process(self, lot: Lot, steps: list):
        lot.total_steps  = len(steps)
        lot.current_step = 0

        for step in steps:
            stn_name = step["station"]
            if stn_name not in self.stations:
                lot.current_step += 1
                continue

            lot.current_station = stn_name
            lot.current_step   += 1

            yield self.env.process(
                self.stations[stn_name].process(
                    lot,
                    step["ptime"],
                    step["setup"],
                    step["setup_cost"],
                    step["proc_unit"],
                )
            )

        # 완료
        lot.finish_time     = self.env.now
        lot.current_station = "DONE"
        ct = lot.finish_time - lot.start_time
        ok = (lot.due_date is None) or (lot.finish_time <= lot.due_date)

        self.kpi_tracker["completed"]    += 1
        self.kpi_tracker["cycle_times"].append(ct)
        self.kpi_tracker["ontime_count"] += int(ok)

        if lot in self.active_lots:
            self.active_lots.remove(lot)
        self.completed_lots.append(lot)

    # ── Lot 투입 컨트롤러 ────────────────────────────────────────────
    def _release_controller(self):
        """order.txt 각 행 → 독립 반복 투입 프로세스 등록"""
        BASE_WIP_LIMIT = self.overrides.get("wip_limit", 3000)
        orders    = self.data.get("orders", pd.DataFrame())
        rkeys     = list(self.route_steps.keys())

        if orders.empty or not rkeys:
            yield self.env.timeout(0)
            return

        col = {c.upper().strip(): c for c in orders.columns}

        for _, row in orders.iterrows():
            lot_name  = _ss(row.get(col.get('LOT', ''), 'LOT'))
            part      = _ss(row.get(col.get('PART', ''), 'part_1'))
            priority  = int(_sf(row.get(col.get('PRIOR', ''), 10), 10))
            wafers    = int(_sf(row.get(col.get('PIECES', ''), 25), 25))
            start_min = _sf(row.get(col.get('START_MIN', ''), 0), 0.0)
            due_min   = _sf(row.get(col.get('DUE_MIN', ''), 99999), 99999.0)
            repeat    = _sf(row.get(col.get('REPEAT', ''), 258.46), 258.46)

            route_key = self._find_route(part, rkeys)

            self.env.process(
                self._repeat_release(
                    lot_name, part, priority, wafers,
                    start_min, due_min, repeat,
                    route_key, BASE_WIP_LIMIT
                )
            )

        yield self.env.timeout(0)

    def _get_dynamic_wip_limit(self, base_limit: int) -> int:
        """
        [수문 제어 원리] 조건부 WIP 상한 적용
        특정 구역(Litho, Implant 등)의 Down 설비가 많으면 전체 투입을 억제함.
        """
        current_wip_limit = base_limit
        
        # 구역별 상태 집계
        area_down_counts = {}
        for name, stn in self.stations.items():
            if stn.is_down:
                area = self._stn_area.get(name, "Unknown")
                area_down_counts[area] = area_down_counts.get(area, 0) + 1
        
        # 임계값 설정
        critical_areas = ["Litho", "Implant", "Diffusion"]
        max_down_impact = 0
        culprit_area = ""
        for area in critical_areas:
            down_count = area_down_counts.get(area, 0)
            if down_count >= 2: # 임계값: 2대 이상 고장 시
                max_down_impact = max(max_down_impact, 0.2)
                culprit_area = area
            elif down_count >= 1: # 1대 고장 시 소폭 억제
                max_down_impact = max(max_down_impact, 0.05)
                if not culprit_area: culprit_area = area
                
        if max_down_impact > 0:
            current_wip_limit = int(base_limit * (1.0 - max_down_impact))
            if not self.gate_control_active:
                self.gate_control_active = True
                msg = f"🚧 [Gate Control] Active: {culprit_area} failure detected. WIP Limit throttled to {current_wip_limit}."
                self.gate_logs.append(msg)
        else:
            if self.gate_control_active:
                self.gate_control_active = False
                msg = f"🔓 [Gate Control] Deactivated: All critical areas recovered. WIP Limit restored to {base_limit}."
                self.gate_logs.append(msg)
            
        return current_wip_limit

    def _find_route(self, part: str, rkeys: list) -> str:
        """Product_1 -> part_1 -> route_steps 키 탐색"""
        # "Product_1" -> "1", "part_1" -> "1"
        num = part.split('_')[-1]
        
        # 우선순위 1: part_{숫자} 형태 (가장 표준)
        target = f"part_{num}"
        if target in self.route_steps:
            return target
            
        # 우선순위 2: r_{숫자} 형태 (일부 txt 파일 내부 이름)
        target_r = f"r_{num}"
        for rk in rkeys:
            if rk.endswith(f"_{num}"):
                return rk
                
        # 우선순위 3: 입력값 그대로
        if part in self.route_steps:
            return part
            
        return rkeys[0]

    def _repeat_release(self, lot_name, part, priority, wafers,
                         start_min, due_min, repeat_interval,
                         route_key, wip_limit):
        """단일 Lot 타입을 repeat_interval 간격으로 반복 투입"""
        delay = max(0.0, start_min - self.env.now)
        if delay > 0:
            yield self.env.timeout(delay)

        counter  = 0
        lot_due_duration = max(1.0, due_min - start_min)

        while True:
            # WIP 상한 초과 시 대기 (수문 제어: 동적 상한 적용)
            current_limit = self._get_dynamic_wip_limit(wip_limit)
            while len(self.active_lots) >= current_limit:
                yield self.env.timeout(repeat_interval)
                # 대기 중에도 상한선은 변할 수 있으므로 갱신
                current_limit = self._get_dynamic_wip_limit(wip_limit)

            lot = Lot(
                lot_id    = f"{lot_name}_{counter:05d}",
                part      = part,
                start_time= self.env.now,
                priority  = priority,
                wafers    = wafers,
                due_date  = self.env.now + lot_due_duration,
            )
            # CQT 데드라인 설정 (시뮬레이션: 전체 공정 시간의 약 80% 지점을 데드라인으로 설정)
            lot.cqt_deadline = self.env.now + (lot_due_duration * 0.8)
            
            counter += 1
            self.active_lots.append(lot)
            self.release_events.append(self.env.now)
            # 최근 1시간(60분) 데이터만 유지
            if len(self.release_events) > 1000:
                self.release_events = [t for t in self.release_events if t > self.env.now - 60]

            self.env.process(
                self._lot_process(lot, self.route_steps[route_key])
            )
            yield self.env.timeout(repeat_interval)

    def get_snapshot(self) -> dict:
        """강화학습 또는 분석을 위한 핵심 피처 추출"""
        now = self.env.now
        
        # 1. 시뮬레이션 시간
        snapshot = {
            "time": now,
            "stations": {},
            "release_rate": 0.0
        }
        
        # 2~4. 설비별 정보 (대기열, 상태, 고장 후 경과 시간)
        for name, stn in self.stations.items():
            # 상태 매핑: down -> Down, busy/setup -> Running, idle -> Idle
            raw_state = stn.state
            if raw_state == "down":
                status = "Down"
            elif raw_state in ("busy", "setup"):
                status = "Running"
            else:
                status = "Idle"

            snapshot["stations"][name] = {
                "queue_size": stn.queue_size,
                "status": status,
                "time_since_last_failure": stn.time_since_last_failure
            }
            
        # 5. 현재 투입률 (최근 60분간 투입된 로트 수 -> 시간당 환산)
        recent_releases = [t for t in self.release_events if t > now - 60]
        self.release_events = recent_releases # 관리용 리스트 갱신
        
        if now > 0:
            window = min(60.0, now)
            snapshot["release_rate"] = round(len(recent_releases) * (60.0 / window), 2)
        else:
            snapshot["release_rate"] = 0.0
        
        return snapshot

    # ── UI 상태 추출 ─────────────────────────────────────────────────
    def update_ui_state(self) -> dict:
        stn_states = []
        area_stats: dict[str, dict] = {}
        gnn_logs = []

        for name, stn in self.stations.items():
            state = stn.state
            area  = self._stn_area.get(name, "Dry_Etch")
            stn_states.append({"id": name, "state": state,
                                "util": stn.utilization, "area": area})
            
            # GNN/CQT 로그 수집 (최근 로그 위주)
            if hasattr(stn, 'action_logs') and stn.action_logs:
                gnn_logs.extend(stn.action_logs)
                stn.action_logs = [] # 읽은 로그 비우기

            if area not in area_stats:
                area_stats[area] = {"busy":0,"down":0,"setup":0,"idle":0,"total":0, "wip": 0}
            area_stats[area][state] += 1
            area_stats[area]["total"] += 1

        # 구역별 WIP 및 Utilization 합산
        area_utils_sum: dict[str, list] = {}
        for name, stn in self.stations.items():
            area = self._stn_area.get(name, "Dry_Etch")
            if area not in area_utils_sum: area_utils_sum[area] = []
            area_utils_sum[area].append(stn.utilization)

        # CQT 위반 현황 계산
        cqt_violations = 0
        urgent_lots = []
        for lot in self.active_lots:
            area = self._stn_area.get(lot.current_station, "Unknown")
            if area in area_stats:
                area_stats[area]["wip"] += 1
            
            if lot.cqt_deadline:
                if self.env.now > lot.cqt_deadline:
                    cqt_violations += 1
                elif lot.cqt_deadline - self.env.now < 120: # 2시간 이내 긴급
                    urgent_lots.append({
                        "id": lot.id,
                        "area": area,
                        "rem": round(lot.cqt_deadline - self.env.now, 1)
                    })

        # 구역별 평균 가동률 계산
        for area, utils in area_utils_sum.items():
            if area in area_stats:
                area_stats[area]["avg_util"] = round(float(np.mean(utils)), 1)

        lot_info = []
        for lot in self.active_lots[:50]:
            cr = 999.0
            if lot.due_date and self.env.now > 0:
                rem = max(1, lot.total_steps - lot.current_step)
                cr  = round((lot.due_date - self.env.now) / rem, 2)
            lot_info.append({
                "id":      lot.id,
                "part":    lot.part,
                "station": lot.current_station or "waiting",
                "step":    lot.current_step,
                "total":   lot.total_steps,
                "cr":      cr,
                "tardy":   bool(lot.due_date and self.env.now > lot.due_date),
                "priority":lot.priority,
            })

        completed  = self.kpi_tracker["completed"]
        cts        = self.kpi_tracker["cycle_times"]
        avg_ct     = round(float(np.mean(cts)) / 60.0, 1) if cts else 0.0
        ontime_pct = round(self.kpi_tracker["ontime_count"] / completed * 100, 1) \
                     if completed > 0 else 0.0
        down_count = sum(1 for s in stn_states if s["state"] == "down")
        wip        = len(self.active_lots)
        tick       = int(self.env.now)
        current_wip_limit = self._get_dynamic_wip_limit(3000) # 기본 3000 기준 (여기서 로그 발생 가능)

        self.wip_history.append({"tick": tick, "wip": wip, "limit": current_wip_limit})
        self.kpi_history.append({"tick": tick, "ct": avg_ct, "ontime": ontime_pct})
        if len(self.wip_history) > 60: self.wip_history = self.wip_history[-60:]
        if len(self.kpi_history) > 60: self.kpi_history = self.kpi_history[-60:]

        # XGBoost 병목 예측
        # 현재 상태를 snapshot 형태로 구성
        snapshot = {
            "area_stats": area_stats,
            "wip": len(self.active_lots),
            "tick": self.env.now
        }
        xgb_probs, xgb_logs = self.xgb_predictor.predict(snapshot)
        
        # 모든 로그 통합 (GNN + XGB + Gate Control)
        gnn_logs.extend(xgb_logs)
        gnn_logs.extend(self.gate_logs)
        self.gate_logs = [] # 읽은 로그 비우기

        return {
            "tick": tick, "wip": wip,
            "stations": stn_states, "area_stats": area_stats,
            "lot_info": lot_info,
            "gnn_logs": gnn_logs,
            "xgb_probs": xgb_probs, # 구역별 병목 확률 추가
            "kpi": {"completed": completed, "avg_ct": avg_ct,
                    "ontime_pct": ontime_pct, "down_count": down_count},
            "cqt": {
                "violations": cqt_violations,
                "urgent_count": len(urgent_lots),
                "urgent_list": urgent_lots[:10]
            },
            "wip_history": self.wip_history[-30:],
            "kpi_history": self.kpi_history[-30:],
        }

    def run_step(self, until: int) -> dict:
        self.env.run(until=until)
        return self.update_ui_state()

    def force_station_down(self, station_name: str, duration: float):
        if station_name in self.stations:
            self.env.process(self._manual_down(self.stations[station_name], duration))

    def _manual_down(self, stn: AdvancedStation, duration: float):
        stn.is_down = True
        stn.stats["down_time"] += duration
        yield self.env.timeout(duration)
        stn.is_down = False

    def set_lot_priority(self, lot_id: str, new_priority: int) -> bool:
        for lot in self.active_lots:
            if lot.id == lot_id:
                lot.priority = new_priority
                return True
        return False

    def get_summary(self) -> dict:
        total = len(self.stations)
        down  = sum(1 for s in self.stations.values() if s.state == "down")
        busy  = sum(1 for s in self.stations.values() if s.state == "busy")
        return {
            "total_stations": total,
            "busy": busy, "down": down, "idle": total - busy - down,
            "wip": len(self.active_lots),
            "completed": self.kpi_tracker["completed"],
        }