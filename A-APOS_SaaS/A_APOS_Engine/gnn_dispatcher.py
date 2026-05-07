"""
gnn_dispatcher.py — GNN 기반 실시간 디스패칭 인터페이스 (Placeholder)

이 모듈은 SimPy 엔진과 GNN 모델 사이의 가교 역할을 합니다.
1. SimPy 상태를 그래프(Subgraph)로 변환
2. GNN 모델 호출 (Score 계산)
3. 최적의 Lot 선택
"""
import random

class GNNDispatcher:
    def __init__(self):
        # 향후 실제 GNN 모델 로드 (PyTorch/DGL 등)
        self.model = None

    def get_subgraph(self, station, waiting_lots):
        """
        현재 설비와 대기열의 상태를 그래프 구조로 추출.
        - Nodes: 설비(1개), Lot(N개), 각 Lot의 현재 공정(Operation)
        - Edges: 'waiting_at', 'requires_setup', 'due_date_proximity' 등
        """
        nodes = []
        edges = []
        
        # 1. 설비 노드
        nodes.append({
            "type": "station",
            "id": station.name,
            "state": station.state,
            "setup": station.current_setup
        })
        
        # 2. Lot 노드
        for lot in waiting_lots:
            # CQT Urgency 계산
            urgency = lot.get_cqt_urgency(station.env.now, station.mean_ptime)
            
            nodes.append({
                "type": "lot",
                "id": lot.id,
                "priority": lot.priority,
                "wafers": lot.wafers,
                "setup_req": lot.setup_req,
                "due_date": lot.due_date,
                "cqt_urgency": urgency  # GNN 피처로 추가
            })
            
            # 3. 에지 (Lot -> Station)
            edges.append((lot.id, station.name, "waiting"))
            
        return {"nodes": nodes, "edges": edges}

    def compute_scores(self, station, waiting_lots):
        """
        GNN 모델을 사용하여 각 Lot에 대한 우선순위 점수를 계산합니다.
        지금은 GNN 로직을 시뮬레이션하기 위한 가중치 기반 랜덤/휴리스틱 점수를 반환합니다.
        """
        if not waiting_lots:
            return {}

        subgraph = self.get_subgraph(station, waiting_lots)
        
        # [Placeholder] 실제 GNN 추론 코드 (예시)
        # scores = self.model(subgraph)
        
        # 시뮬레이션용 휴리스틱 점수 계산:
        # 1. 우선순위(priority)가 낮을수록(값이 작을수록 중요) 점수 높음
        # 2. 납기(due_date)가 가까울수록 점수 높음
        # 3. 현재 설비 셋업(setup)과 일치하면 점수 가점
        
        scores = {}
        for lot in waiting_lots:
            base_score = 100 - lot.priority  # 기본 우선순위
            
            # 납기 긴박도 (Critical Ratio 스타일)
            # 여기서는 간단히 0~50점 추가
            due_bonus = random.uniform(0, 20) 
            
            # 셋업 일치 가점
            setup_bonus = 30 if lot.setup_req == station.current_setup else 0
            
            # CQT Urgency 가점 (EWS 스타일)
            # urgency가 낮을수록(데드라인 임박) 점수 대폭 상승
            urgency = lot.get_cqt_urgency(station.env.now, station.mean_ptime)
            urgency_bonus = 0
            if urgency < 2.0:   # 매우 긴급
                urgency_bonus = 100
                station.action_logs.append(f"🔥 CQT CRITICAL: Lot {lot.id} (Urgency: {urgency:.2f})")
            elif urgency < 5.0: # 긴급
                urgency_bonus = 50
                station.action_logs.append(f"⚠️ CQT URGENT: Lot {lot.id} (Urgency: {urgency:.2f})")
            elif urgency < 10.0: # 주의
                urgency_bonus = 20
            
            scores[lot.id] = base_score + due_bonus + setup_bonus + urgency_bonus
            
        return scores

    def select_best_lot(self, station, waiting_lots):
        """최적의 Lot 하나를 선택하여 반환"""
        if not waiting_lots:
            return None
            
        scores = self.compute_scores(station, waiting_lots)
        best_lot = max(waiting_lots, key=lambda l: scores.get(l.id, 0))
        
        return best_lot, scores
