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
    def __init__(self, env: simpy.Environment, data: dict):
        self.env            = env
        self.data           = data
        self.stations: dict[str, AdvancedStation] = {}
        self.active_lots:    list = []
        self.completed_lots: list = []
        self.kpi_tracker = {
            "completed":    0,
            "cycle_times":  [],
            "ontime_count": 0,
        }
        self.wip_history: list = []
        self.kpi_history: list = []
        self._stn_area:   dict[str, str] = {}
        self.release_events: list = []  # 로트 투입 시점 기록 (최근 투입률 계산용)

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

        for name, cfg in stn_cfg.items():
            # 실제 설비 대수를 capacity로 반영 (없으면 기본값 1)
            capacity = tool_capacity.get(name, 1)
            self.stations[name] = AdvancedStation(
                env, name,
                capacity=capacity,
                is_batch=cfg["is_batch"],
                batch_min_wafers=cfg["batch_min_wafers"],
                batch_max_wafers=cfg["batch_max_wafers"],
            )

        # ── 3. 고장 프로세스 (DS3, DS4) ─────────────────────────────
        if data.get("downs") is not None:
            area_bd = self._parse_downcal(data["downs"])
            for stn_name, stn_obj in self.stations.items():
                area = self._stn_area.get(stn_name, "Dry_Etch")
                mttf, mttr = area_bd.get(area, BREAKDOWN_TABLE.get(area, (10080, 200)))
                env.process(failure_process(env, stn_obj, mttf, mttr))

        # ── 4. Lot 투입 등록 ─────────────────────────────────────────
        if self.route_steps:
            env.process(self._release_controller())

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
        WIP_LIMIT = 3000
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
                    route_key, WIP_LIMIT
                )
            )

        yield self.env.timeout(0)

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
            # WIP 상한 초과 시 대기
            while len(self.active_lots) >= wip_limit:
                yield self.env.timeout(repeat_interval)

            lot = Lot(
                lot_id    = f"{lot_name}_{counter:05d}",
                part      = part,
                start_time= self.env.now,
                priority  = priority,
                wafers    = wafers,
                due_date  = self.env.now + lot_due_duration,
            )
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

        for name, stn in self.stations.items():
            state = stn.state
            area  = self._stn_area.get(name, "Dry_Etch")
            stn_states.append({"id": name, "state": state,
                                "util": stn.utilization, "area": area})
            if area not in area_stats:
                area_stats[area] = {"busy":0,"down":0,"setup":0,"idle":0,"total":0}
            area_stats[area][state] += 1
            area_stats[area]["total"] += 1

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

        self.wip_history.append({"tick": tick, "wip": wip})
        self.kpi_history.append({"tick": tick, "ct": avg_ct, "ontime": ontime_pct})
        if len(self.wip_history) > 60: self.wip_history = self.wip_history[-60:]
        if len(self.kpi_history) > 60: self.kpi_history = self.kpi_history[-60:]

        return {
            "tick": tick, "wip": wip,
            "stations": stn_states, "area_stats": area_stats,
            "lot_info": lot_info,
            "kpi": {"completed": completed, "avg_ct": avg_ct,
                    "ontime_pct": ontime_pct, "down_count": down_count},
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