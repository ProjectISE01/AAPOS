"""
gnn_dispatcher.py — GNN 기반 실시간 디스패칭 인터페이스 (Placeholder)

이 모듈은 SimPy 엔진과 GNN 모델 사이의 가교 역할을 합니다.
1. SimPy 상태를 그래프(Subgraph)로 변환
2. GNN 모델 호출 (Score 계산)
3. 최적의 Lot 선택
"""
import random

class GNNDispatcher:
    def __init__(self, policy="GNN"):
        # 정책 설정: "FIFO", "EDD", "CR", "GNN"
        self.policy = policy
        self.model = None

    def compute_scores(self, station, waiting_lots):
        """
        설정된 정책에 따라 각 Lot에 대한 우선순위 점수를 계산합니다.
        점수가 높을수록 우선순위가 높음.
        """
        if not waiting_lots:
            return {}

        now = station.env.now
        scores = {}

        for lot in waiting_lots:
            if self.policy == "FIFO":
                # 먼저 도착한 순서 (도착 시간의 역순)
                scores[lot.id] = 1000000 - lot.start_time
            
            elif self.policy == "EDD":
                # 납기일이 빠른 순서 (납기일의 역순)
                due = lot.due_date if lot.due_date else 999999
                scores[lot.id] = 1000000 - due
            
            elif self.policy == "CR":
                # Critical Ratio = (Due Date - Now) / Remaining Process Time
                # 여기선 간단히 (Due Date - Now)의 역순으로 계산 (작을수록 좋음)
                due = lot.due_date if lot.due_date else 999999
                rem_time = max(1, (lot.total_steps - lot.current_step) * station.mean_ptime)
                cr = (due - now) / rem_time
                scores[lot.id] = 1000 - cr
            
            else: # "GNN" (Heuristic + AI Hybrid)
                base_score = 100 - lot.priority
                due_bonus = 20 if (lot.due_date and lot.due_date - now < 500) else 0
                setup_bonus = 30 if lot.setup_req == station.current_setup else 0
                
                # CQT Urgency 가점
                urgency = lot.get_cqt_urgency(now, station.mean_ptime)
                urgency_bonus = 0
                if urgency < 2.0: urgency_bonus = 100
                elif urgency < 5.0: urgency_bonus = 50
                
                scores[lot.id] = base_score + due_bonus + setup_bonus + urgency_bonus
            
        return scores

    def select_best_lot(self, station, waiting_lots):
        """최적의 Lot 하나를 선택하여 반환"""
        if not waiting_lots:
            return None
            
        scores = self.compute_scores(station, waiting_lots)
        best_lot = max(waiting_lots, key=lambda l: scores.get(l.id, 0))
        
        return best_lot, scores
