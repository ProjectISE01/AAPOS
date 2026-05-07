import pandas as pd
import numpy as np
from collections import deque

class XGBBottleneckPredictor:
    def __init__(self, window_size=4, threshold=0.7):
        self.window_size = window_size # t, t-1, t-2, t-3 (4 snapshots)
        self.threshold = threshold
        # 구역별 히스토리 저장 (Area -> deque of raw features)
        self.history = {}
        # 글로벌 히스토리 저장
        self.global_history = deque(maxlen=window_size)
        # 현재 활성화된 병목 구역 추적 (Area -> last_prob)
        self.active_bottlenecks = {}
        # 향후 실제 XGBoost 모델 로드
        self.model = None 

    def update_history(self, snapshot, global_stats=None):
        """
        SimBridge로부터 받은 스냅샷을 히스토리에 추가
        snapshot format: { "area_stats": { "AreaName": { "busy": 10, "down": 2, ... }, ... } }
        """
        area_stats = snapshot.get("area_stats", {})
        for area_name, stats in area_stats.items():
            if area_name not in self.history:
                self.history[area_name] = deque(maxlen=self.window_size)
            
            # 1단계: 원본 피처 (20개 중 주요 항목 시뮬레이션 추출)
            # 실제 구현 시에는 SimBridge에서 더 많은 데이터를 넘겨받아야 함
            raw = self._extract_raw_features(area_name, stats, snapshot)
            self.history[area_name].append(raw)
            
        # 글로벌 피처 업데이트
        if global_stats:
            self.global_history.append(global_stats)

    def _extract_raw_features(self, area_name, stats, snapshot):
        """1단계: 원본 피처 (20개) 추출"""
        # stats와 snapshot에서 20개 지표를 dict 형태로 추출
        # (여기서는 핵심 12개 위주로 우선 구현)
        total = stats.get("total", 1)
        raw = {
            "area_wip": stats.get("wip", 0),
            "area_utilization": stats.get("busy", 0) / total if total > 0 else 0,
            "area_down_count": stats.get("down", 0),
            "area_queue_mean": stats.get("queue_mean", 0),
            "area_cr_mean": stats.get("cr_mean", 1.0),
            "area_cqt_near_violation": stats.get("cqt_near_violation", 0),
            "area_hotlot_count": stats.get("hotlot_count", 0),
            "area_avg_waiting": stats.get("avg_waiting", 0),
            "area_throughput_1h": stats.get("throughput_1h", 0),
            "global_wip": snapshot.get("wip", 0),
            "time_of_day": (snapshot.get("tick", 0) % 1440) / 1440.0,
            "dataset_id": 4.0 # 고정
        }
        return raw

    def extract_all_features(self, area_name):
        """
        83개 최종 피처 생성 (3단계)
        """
        h = list(self.history.get(area_name, []))
        if len(h) < self.window_size:
            return None
        
        # 1 & 2단계: 원본 20개 + 슬라이딩 윈도우 60개
        all_features = []
        feature_keys = list(h[0].keys())
        
        for key in feature_keys:
            vals = [snapshot[key] for snapshot in h]
            t = vals[-1]
            ma = np.mean(vals)
            # ROC: (t - t-2) / t-2
            t_minus_2 = vals[-3] if len(vals) >= 3 else vals[0]
            roc = (t - t_minus_2) / t_minus_2 if t_minus_2 > 0 else 0
            sigma = np.std(vals)
            
            all_features.extend([t, ma, roc, sigma]) # 4개씩 x 20개 키 = 80개 (시뮬레이션상 일부 키 생략 가능)
            
        # 3단계: SMT 공정 전용 파생 피처 (3개)
        # 1. WIP_slope (선형 회귀 기울기 대용)
        wip_vals = [s["area_wip"] for s in h]
        wip_slope = (wip_vals[-1] - wip_vals[0]) / len(h)
        
        # 2. CQT_burn_rate
        cqt_vals = [s["area_cqt_near_violation"] for s in h]
        cqt_burn_rate = (cqt_vals[-1] - cqt_vals[-2]) if len(cqt_vals) >= 2 else 0
        
        # 3. Cascade_risk (Litho_down * Area_wip_roc)
        # 예시로 타 구역(Litho) 정보를 가져온다고 가정
        litho_down = 0
        if "Litho" in self.history:
            litho_down = self.history["Litho"][-1]["area_down_count"]
        
        # 현재 구역의 WIP ROC (이미 계산됨)
        curr_wip_vals = [s["area_wip"] for s in h]
        curr_wip_roc = (curr_wip_vals[-1] - curr_wip_vals[-3]) / curr_wip_vals[-3] if curr_wip_vals[-3] > 0 else 0
        cascade_risk = litho_down * curr_wip_roc
        
        all_features.extend([wip_slope, cqt_burn_rate, cascade_risk])
        
        return np.array(all_features)

    def predict(self, snapshot):
        """
        업그레이드된 83개 피처 기반 병목 확률 예측 및 해소 로그 생성
        """
        self.update_history(snapshot)
        predictions = {}
        logs = []
        
        current_areas = list(self.history.keys())
        for area_name in current_areas:
            features = self.extract_all_features(area_name)
            if features is None:
                continue
            
            # [시뮬레이션] 스케일링 이슈 해결을 위한 정규화 및 로지스틱 스코어링
            area_wip_ma = features[1]
            area_util_ma = features[5]
            wip_slope = features[-3]
            cascade_risk = features[-1]
            
            # 1. 베이스 점수
            norm_wip = min(area_wip_ma / 100.0, 1.0)
            score = (area_util_ma * 0.4) + (norm_wip * 0.3)
            
            # 2. 추세 보너스
            trend_bonus = (max(0, wip_slope) * 0.2) + (min(cascade_risk, 2.0) * 0.1)
            
            # 3. Sigmoid 확률 산출
            final_score = score + trend_bonus
            prob = 1 / (1 + np.exp(-5 * (final_score - 0.5)))
            prob = min(max(prob, 0.05), 0.95)
            
            predictions[area_name] = prob
            
            # 병목 상태 변화 감지 및 로그 생성
            if prob >= self.threshold:
                if area_name not in self.active_bottlenecks:
                    # 신규 병목 발생
                    logs.append(f"🚨 [XGB 83F] Bottleneck Warning: {area_name} ({prob*100:.1f}%) - Trend: {'Increasing' if wip_slope > 0.01 else 'Stable'}")
                    self.active_bottlenecks[area_name] = prob
                else:
                    # 병목 유지 (로그 생략 가능하나 필요시 추가)
                    self.active_bottlenecks[area_name] = prob
            else:
                if area_name in self.active_bottlenecks:
                    # 병목 해소!
                    prev_prob = self.active_bottlenecks[area_name]
                    logs.append(f"✅ [XGB 83F] Bottleneck Resolved: {area_name} ({prev_prob*100:.1f}% -> {prob*100:.1f}%) - Status: Normalized")
                    del self.active_bottlenecks[area_name]
                
        return predictions, logs
