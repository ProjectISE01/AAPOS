"""
factory_engine.py — A-APOS SimPy 핵심 엔진 (Final)

SMT 2020 데이터 기반 설계 원칙:
- BATCH MINIMUM/MAXIMUM: Wafers 기준 (25 wafers/lot)
- PROCESSING UNIT: Batch=설비당 1회 처리, Wafer=웨이퍼별, Lot=lot 단위
- 고장: is_down 플래그만 사용 (Resource interrupt 없음 → Preempted 에러 방지)
- 배치: 리더/멤버 구조, 리더만 설비 점유
"""
import simpy
import random


class Lot:
    def __init__(self, lot_id, part, start_time, priority,
                 wafers=25, setup_req="", due_date=None):
        self.id              = lot_id
        self.part            = part
        self.start_time      = start_time
        self.priority        = priority
        self.wafers          = wafers       # WAFERS PER LOT (보통 25)
        self.setup_req       = setup_req
        self.due_date        = due_date
        self.current_step    = 0
        self.total_steps     = 0
        self.current_station = None
        self.finish_time     = None
        self.wait_event      = None
        self.batch_done_ev   = None   # 배치 완료 이벤트 (리더가 심어줌)
        self.is_leader       = False  # 배치 리더 여부

    @property
    def is_tardy(self) -> bool:
        if self.due_date is None or self.finish_time is None:
            return False
        return self.finish_time > self.due_date


class AdvancedStation:
    """
    배치 처리 흐름:
    1. Lot이 도착 → batch_queue에 추가
    2. 총 wafers >= batch_min_wafers → 즉시 배치 출발
    3. BATCH_WAIT_MAX 초과 → 강제 출발
    4. 리더(첫 배치 채운 Lot)가 설비 점유 → ptime 처리
    5. 멤버는 리더 완료 이벤트 대기 → 통과

    고장 처리:
    - is_down 플래그만 사용
    - 처리 시작 전 is_down 체크 → 복구까지 대기
    """
    BATCH_WAIT_MAX = 200.0  # 분

    def __init__(self, env, name, capacity=1,
                 is_batch=False, batch_min_wafers=1, batch_max_wafers=1):
        self.env               = env
        self.name              = name
        self.res               = simpy.Resource(env, capacity=capacity)
        self.is_batch          = is_batch
        self.batch_min_wafers  = max(1, batch_min_wafers)
        self.batch_max_wafers  = max(1, batch_max_wafers)
        self.batch_queue       = []
        self.batch_done_event  = None
        self.is_down           = False
        self.last_failure_time  = 0.0
        self.last_recovery_time = 0.0
        self.current_setup     = None
        self.stats = {
            "util_time":      0.0,
            "setup_time":     0.0,
            "down_time":      0.0,
            "lots_processed": 0,
        }

    @property
    def state(self) -> str:
        if self.is_down:
            return "down"
        if self.res.count > 0:
            return "busy"
        if self.is_batch and self.batch_queue:
            return "setup"
        return "idle"

    @property
    def queue_size(self) -> int:
        """현재 대기 중인 로트 수 (배치 큐 + 리소스 대기 큐)"""
        return len(self.batch_queue) + len(self.res.queue)

    @property
    def time_since_last_failure(self) -> float:
        """최근 고장 발생 후 경과 시간"""
        return self.env.now - self.last_failure_time

    @property
    def utilization(self) -> float:
        now = self.env.now
        return round(self.stats["util_time"] / now * 100, 1) if now > 0 else 0.0

    @property
    def queued_wafers(self) -> int:
        return sum(lot.wafers for lot in self.batch_queue)

    def _release_batch(self):
        """배치 큐 모두 깨우고 초기화"""
        waiting = list(self.batch_queue)
        self.batch_queue = []
        for l in waiting:
            if l.wait_event and not l.wait_event.triggered:
                l.wait_event.succeed()

    def process(self, lot, ptime: float, setup_req: str,
                setup_cost: float, proc_unit: str = "Lot"):
        """
        proc_unit:
          'Batch' → ptime 그대로 (배치 전체 처리시간)
          'Lot'   → ptime 그대로
          'Wafer' → ptime × lot.wafers
        """
        # 고장 복구 대기
        while self.is_down:
            yield self.env.timeout(30)

        # 처리시간 계산
        if proc_unit == "Wafer":
            actual_ptime = max(0.01, ptime * lot.wafers)
        else:
            actual_ptime = max(0.01, ptime)

        # 셋업 변경
        if setup_req and setup_req != self.current_setup and setup_cost > 0:
            self.stats["setup_time"] += setup_cost
            yield self.env.timeout(setup_cost)
            self.current_setup = setup_req

        if self.is_batch:
            yield self.env.process(self._batch_proc(lot, actual_ptime))
        else:
            yield self.env.process(self._single_proc(lot, actual_ptime))

    def _single_proc(self, lot, ptime: float):
        """비배치 개별 처리"""
        with self.res.request() as req:
            yield req
            while self.is_down:
                yield self.env.timeout(30)
            start = self.env.now
            lot.current_station = self.name
            yield self.env.timeout(ptime)
            self.stats["util_time"]      += self.env.now - start
            self.stats["lots_processed"] += 1

    def _batch_proc(self, lot, ptime: float):
        """
        배치 처리 — 독립 done_event 구조

        핵심 수정:
        - 배치 출발 시 done_event를 생성하고 모든 멤버의 lot.batch_done_ev에 직접 심음
        - 멤버는 자신의 lot.batch_done_ev를 참조 → 인스턴스 변수 덮어쓰기 문제 제거
        """
        self.batch_queue.append(lot)
        lot.wait_event    = self.env.event()
        lot.batch_done_ev = None   # 배치 출발 시 리더가 채워줌
        is_leader = False

        if self.queued_wafers >= self.batch_min_wafers:
            self._release_batch_with_event(lot)
        else:
            timeout_ev = self.env.timeout(self.BATCH_WAIT_MAX)
            yield lot.wait_event | timeout_ev
            if not lot.wait_event.triggered:
                self._release_batch_with_event(lot)

        # 리더 여부: 리더는 batch_done_ev를 직접 완료시켜야 하는 주체
        # _release_batch_with_event 안에서 lot.is_leader 플래그 설정
        is_leader = getattr(lot, 'is_leader', False)

        lot.current_station = self.name

        if is_leader:
            # 리더: 설비 점유 → 처리 → 자신의 done_ev 완료
            with self.res.request() as req:
                yield req
                while self.is_down:
                    yield self.env.timeout(30)
                start = self.env.now
                yield self.env.timeout(ptime)
                self.stats["util_time"]      += self.env.now - start
                self.stats["lots_processed"] += 1

            # 자신의 done_ev 완료 → 멤버들이 깨어남
            if lot.batch_done_ev and not lot.batch_done_ev.triggered:
                lot.batch_done_ev.succeed()
        else:
            # 멤버: 자신에게 심긴 done_ev 대기
            if lot.batch_done_ev and not lot.batch_done_ev.triggered:
                yield lot.batch_done_ev
            else:
                yield self.env.timeout(0)
            self.stats["lots_processed"] += 1

    def _release_batch_with_event(self, trigger_lot=None):
        """
        배치 큐 전체를 깨우고,
        이번 배치 전용 done_event를 생성해 모든 Lot에 심어줌.
        큐의 첫 번째 Lot이 리더 → is_leader=True 설정.
        trigger_lot: 타임아웃이나 수량 충족을 감지한 Lot (리더 후보)
        """
        waiting = list(self.batch_queue)
        self.batch_queue = []

        if not waiting:
            return

        # 이번 배치 전용 done_event (모든 멤버가 공유)
        done_ev = self.env.event()

        # 리더 = 큐의 첫 번째 Lot
        leader = waiting[0]
        leader.is_leader = True

        for l in waiting:
            l.batch_done_ev = done_ev  # 같은 이벤트 공유
            if l is not leader:
                l.is_leader = False
                if l.wait_event and not l.wait_event.triggered:
                    l.wait_event.succeed()

        # 리더 wait_event 트리거 (배치 출발)
        if leader.wait_event and not leader.wait_event.triggered:
            leader.wait_event.succeed()


def failure_process(env, station: AdvancedStation, mttf: float, mttr: float):
    """구역 단위 고장 — is_down 플래그만 사용 (Resource interrupt 없음)"""
    while True:
        yield env.timeout(random.expovariate(1.0 / mttf))
        station.is_down = True
        station.last_failure_time = env.now
        down_dur = random.expovariate(1.0 / mttr)
        station.stats["down_time"] += down_dur
        yield env.timeout(down_dur)
        station.is_down = False
        station.last_recovery_time = env.now