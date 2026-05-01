"""
data_manager.py — SMT 2020 데이터 로더 (Final)

실제 파일 구조 (직접 확인):
- order.txt: LOT, PART, PRIOR, PIECES, START, REPEAT, RPT#, DUE
- route_*.txt: STNFAM, PTIME, PTPER, BATCHMN, BATCHMX, STIME, IGNORE
- tool.txt: STNFAM(설비그룹), NUMTOOLS(대수) 포함
- downcal.txt: 구역별 MTTF, MTTR (DS3, DS4만)

DS별 구조:
  DS1(HVLM):   route_3, route_4
  DS2(LVHM):   route_1~10
  DS3(HVLM_E): route_3, route_4, route_E3
  DS4(LVHM_E): route_1~10, route_E1~E3
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from io import StringIO

BASE_DATE = datetime(2018, 1, 1)


def _read_utf16(file_path: str) -> pd.DataFrame:
    """UTF-16 BOM 파일을 안정적으로 읽어 DataFrame 반환"""
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
    except Exception:
        return pd.DataFrame()

    text = None
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        try:
            text = raw.decode('utf-16')
        except Exception:
            pass

    if text is None:
        for enc in ['utf-8', 'latin1', 'cp949']:
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue

    if text is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(text), sep='\t', on_bad_lines='skip', engine='python')
    except Exception:
        return pd.DataFrame()

    # 컬럼명 정리
    df.columns = [str(c).strip() for c in df.columns]
    # 문자열 값 정리
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'nan': None, 'NaN': None, 'None': None})
    return df


def _date_to_min(val) -> float:
    """날짜 → 기준일로부터 분 변환"""
    if isinstance(val, datetime):
        return max(0.0, (val - BASE_DATE).total_seconds() / 60.0)
    if isinstance(val, str):
        for fmt in ['%m/%d/%y %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%m/%d/%y']:
            try:
                d = datetime.strptime(val.strip(), fmt)
                return max(0.0, (d - BASE_DATE).total_seconds() / 60.0)
            except Exception:
                continue
    return 0.0


class APOSDataManager:
    def __init__(self, base_path="SMT_2020 - Final/AutoSched"):
        self.base_path = base_path
        self.datasets = {
            1: "dataset 1/HVLM_Model/HVLM_Model.asd",
            2: "dataset 2/LVHM_Model/LVHM_Model.asd",
            3: "dataset 3/HVLM_E_Model/HVLM_E_Model.asd",
            4: "dataset 4/LVHM_E_Model/LVHM_E_Model.asd",
        }

    def load_dataset(self, ds_id: int) -> dict:
        path = os.path.join(os.getcwd(), self.base_path, self.datasets[ds_id])
        if not os.path.exists(path):
            raise FileNotFoundError(f"데이터 경로 없음: {path}")

        # ── order.txt ───────────────────────────────────────────────
        orders = self._load_orders(os.path.join(path, "order.txt"))

        # ── route_*.txt ─────────────────────────────────────────────
        routes = {}
        for fname in sorted(os.listdir(path)):
            if fname.startswith("route_") and fname.endswith(".txt"):
                key = fname.replace("route_", "part_").replace(".txt", "")
                df = _read_utf16(os.path.join(path, fname))
                if not df.empty and 'STNFAM' in df.columns:
                    routes[key] = df

        # ── tool.txt — 설비 대수(capacity) ──────────────────────────
        tool_capacity = self._load_tool_capacity(path)

        # ── pmcal.txt ───────────────────────────────────────────────
        pm_path = os.path.join(path, "pmcal.txt")
        pmcal = _read_utf16(pm_path) if os.path.exists(pm_path) else pd.DataFrame()

        # ── setup.txt ───────────────────────────────────────────────
        setup_path = os.path.join(path, "setup.txt")
        setup = _read_utf16(setup_path) if os.path.exists(setup_path) else pd.DataFrame()

        # ── downcal.txt (DS3, DS4만) ─────────────────────────────────
        downs = None
        if ds_id in [3, 4]:
            down_path = os.path.join(path, "downcal.txt")
            if os.path.exists(down_path):
                downs = _read_utf16(down_path)

        # ── 메타데이터 ───────────────────────────────────────────────
        all_tgs      = set()
        bn_candidates = []
        step_counts  = []

        for df in routes.values():
            col = {c.upper(): c for c in df.columns}
            stn_col  = col.get('STNFAM')
            bmin_col = col.get('BATCHMN')
            stime_col= col.get('STIME')

            if stn_col:
                valid = df[df[stn_col].notna()]
                step_counts.append(len(valid))
                all_tgs.update(valid[stn_col].unique())

            if stn_col and (bmin_col or stime_col):
                try:
                    mask = pd.Series(False, index=df.index)
                    if bmin_col:
                        mask |= pd.to_numeric(df[bmin_col], errors='coerce').fillna(0) >= 100
                    if stime_col:
                        mask |= pd.to_numeric(df[stime_col], errors='coerce').fillna(0) >= 60
                    bn_candidates.extend(
                        df.loc[mask, stn_col].dropna().unique().tolist()
                    )
                except Exception:
                    pass

        metadata = {
            "total_parts":     len(routes),
            "total_orders":    len(orders),
            "avg_steps":       int(np.mean(step_counts)) if step_counts else 0,
            "toolgroup_count": len(all_tgs),
            "bn_candidates":   list(set(bn_candidates)),
        }

        return {
            "orders":        orders,
            "routes":        routes,
            "setup":         setup,
            "pmcal":         pmcal,
            "downs":         downs,
            "metadata":      metadata,
            "tool_capacity": tool_capacity,
        }

    def _load_orders(self, order_path: str) -> pd.DataFrame:
        if not os.path.exists(order_path):
            return pd.DataFrame()

        df = _read_utf16(order_path)
        if df.empty:
            return df

        col = {c.upper(): c for c in df.columns}

        # 유효 행 필터
        lot_col  = col.get('LOT')
        part_col = col.get('PART')
        if lot_col and part_col:
            df = df[df[lot_col].notna() & df[part_col].notna()].copy()

        # 시간 변환
        start_col = col.get('START')
        due_col   = col.get('DUE')
        if start_col:
            df['START_MIN'] = df[start_col].apply(_date_to_min)
        else:
            df['START_MIN'] = 0.0

        if due_col:
            df['DUE_MIN'] = df[due_col].apply(_date_to_min)
        else:
            df['DUE_MIN'] = 99999.0

        # 숫자 변환
        for col_name, default in [('PRIOR', 10), ('PIECES', 25), ('REPEAT', 258.46)]:
            c = col.get(col_name)
            if c:
                df[col_name] = pd.to_numeric(df[c], errors='coerce').fillna(default)

        return df.reset_index(drop=True)

    def _load_tool_capacity(self, path: str) -> dict:
        """
        tool.txt에서 Toolgroup별 설비 대수 파싱.
        AutoSched tool.txt 컬럼: STNFAM, STNQTY, STNCAP
        """
        capacity = {}
        tool_path = os.path.join(path, "tool.txt")

        if not os.path.exists(tool_path):
            return capacity

        df = _read_utf16(tool_path)
        if df.empty:
            return capacity

        col = {c.upper().strip(): c for c in df.columns}

        # Toolgroup 컬럼 탐색
        tg_col = (col.get('STNFAM') or col.get('TOOLGROUP') or
                  col.get('TOOLGRP') or col.get('STNGRP'))
        # 설비 수 컬럼 탐색 (STNQTY, NUMTOOLS 등)
        cnt_col = (col.get('STNQTY') or col.get('NUMTOOLS') or 
                   col.get('NUMBER OF TOOLS') or col.get('NUMBER') or 
                   col.get('QTY') or col.get('COUNT'))
        # 개별 설비당 용량 (STNCAP)
        cap_col = col.get('STNCAP')

        if not tg_col:
            # 첫 번째 문자열 컬럼이 TG
            str_cols = [c for c in df.columns if df[c].dtype == object]
            if str_cols:
                tg_col = str_cols[0]
        
        if not cnt_col:
            # 숫자가 포함된 적절한 컬럼 탐색
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                cnt_col = num_cols[0]

        if not (tg_col and cnt_col):
            return capacity

        for _, row in df.iterrows():
            tg = str(row.get(tg_col, '')).strip()
            if not tg or tg in ('nan', 'None', ''):
                continue
            try:
                # STNQTY (설비 대수)
                qty = float(str(row.get(cnt_col, 1)).strip())
                # STNCAP (설비당 동시 처리 가능 Lot 수, 기본 1)
                cap = float(str(row.get(cap_col, 1)).strip()) if cap_col else 1.0
                if np.isnan(cap): cap = 1.0
                
                n = int(qty * cap)
                if n > 0:
                    capacity[tg] = n
            except (TypeError, ValueError):
                pass

        return capacity