"""Event-driven twin-crane steel-plate yard. Requires NumPy and Gymnasium."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from itertools import combinations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

LEFT_START_POS = 0
RIGHT_START_POS = 41

LEFT_STORAGE_BAYS = list(range(1, 14))
LEFT_OUTBOUND_BAYS = list(range(14, 19))
HANDSHAKE_BAYS = [19, 20, 21]
RIGHT_OUTBOUND_BAYS = list(range(22, 27))
RIGHT_STORAGE_BAYS = list(range(27, 41))

LEFT_SIDE = 0
RIGHT_SIDE = 1
LEFT_SAFE_BAY = 18
RIGHT_SAFE_BAY = 22

HANDSHAKE_ACTION_TO_SLOT = {
    19: 0,
    20: 1,
    21: 2,
    42: 3,
    43: 4,
    44: 5,
}
HANDSHAKE_SLOT_TO_ACTION = {
    slot: action for action, slot in HANDSHAKE_ACTION_TO_SLOT.items()
}
HANDSHAKE_SLOT_TO_BAY = {
    0: 19,
    1: 20,
    2: 21,
    3: 19,
    4: 20,
    5: 21,
}


SOURCE_BAYS = tuple(LEFT_STORAGE_BAYS + RIGHT_STORAGE_BAYS)
EPS = 1e-8

CRANE_CAPACITY_TONNES = 20.0


class YardSimulator(gym.Env):
    """Event-driven simulator, with zero-based candidate indices 0..32."""

    metadata = {"render_modes": []}

    def __init__(self, K=16, H=5, rho=0.5, capacity=6):
        super().__init__()
        if not isinstance(K, int) or K not in range(2, 27, 2):
            raise ValueError("K must allow an equal number of active bays on both sides")
        if not isinstance(H, int) or not 1 <= H <= 20 or not 0 <= rho <= 1:
            raise ValueError("Expected 1 <= H <= 20 and 0 <= rho <= 1")
        if not isinstance(capacity, int) or not 1 <= capacity <= 6:
            raise ValueError("Handover capacity must be between 1 and 6")
        self.K_s, self.h, self.rho, self.capacity = K, H, rho, capacity
        self.N = K * H
        self.B_max, self.W_max = 3, 20.0
        self.epsilon_o = self.epsilon_d = 2
        self.move_time_per_bay = 1.0
        self.left_start_pos, self.right_start_pos = 0, 41
        self.left_storage_bays, self.right_storage_bays = LEFT_STORAGE_BAYS, RIGHT_STORAGE_BAYS
        self.left_outbound_bays, self.right_outbound_bays = LEFT_OUTBOUND_BAYS, RIGHT_OUTBOUND_BAYS
        self.handshake_bays = HANDSHAKE_BAYS
        self.handshake_action_to_slot = HANDSHAKE_ACTION_TO_SLOT
        self.handshake_slot_to_action = HANDSHAKE_SLOT_TO_ACTION
        self.handshake_slot_to_bay = HANDSHAKE_SLOT_TO_BAY
        self.action_space = spaces.Discrete(33)
        self.observation_space = spaces.Dict({
            'source': spaces.Box(0., 1., (27,16), np.float32),
            'handover': spaces.Box(0., 1., (6,8), np.float32),
            'quality': spaces.Box(0., 1., (33,7), np.float32),
            'source_mask': spaces.MultiBinary(27),
            'action_mask': spaces.MultiBinary(33),
        })

    @property
    def vertical_cycle(self):
        """Service time for one distinct pickup or putdown stop (Eq. 20)."""
        return 6.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_time = self.current_makespan = 0.0
        self.active_crane = LEFT_SIDE
        self.plates, self.batches = {}, {}
        self.crane_positions = [0, 41]
        self.crane_tasks, self.crane_trajectories = [None, None], [[], []]
        self.remaininfor = None
        self.completed_plate_ids, self.reserved_plate_ids, self.reserved_slot_actions = set(), set(), set()
        self._init_handshake_slots()
        if not options or "instance" not in options:
            raise ValueError("reset requires options={'instance': path_to_json}")
        self._load_instance(options["instance"])
        return self._observe(), {"makespan": 0.0}

    def _load_instance(self, path):
        """Read JSON with a 'stacks' list in source order; each stack has a 'bay'
        and top-first 'plates'. Each plate requires 'target_bay' and 'weight' (tonnes).
        """
        with Path(path).open(encoding="utf-8-sig") as file:
            stacks = json.load(file)["stacks"]
        if not isinstance(stacks, list) or len(stacks) != self.K_s:
            raise ValueError(f"{path}: expected {self.K_s} active source stacks")
        self.origin_bays = []
        self.bay_stacks = {}
        cross_counts = [0, 0]
        left_count = 0
        for stack in stacks:
            bay, plates = stack["bay"], stack["plates"]
            if type(bay) is not int or bay not in SOURCE_BAYS or bay in self.bay_stacks:
                raise ValueError(f"{path}: invalid or duplicate source bay {bay}")
            if not isinstance(plates, list) or len(plates) != self.h:
                raise ValueError(f"{path}: expected {self.h} plates at bay {bay}")
            side = int(bay in RIGHT_STORAGE_BAYS)
            left_count += int(side == LEFT_SIDE)
            self.origin_bays.append(bay)
            self.bay_stacks[bay] = []
            for item in plates:
                target, mass = item["target_bay"], item["weight"]
                if type(target) is not int or target not in LEFT_OUTBOUND_BAYS + RIGHT_OUTBOUND_BAYS:
                    raise ValueError(f"{path}: invalid target bay {target}")
                if type(mass) not in (int, float) or not 0 < mass <= self.W_max:
                    raise ValueError(f"{path}: invalid plate weight {mass}")
                target_side = int(target in RIGHT_OUTBOUND_BAYS)
                cross = side != target_side
                cross_counts[side] += int(cross)
                pid = len(self.plates)
                self.plates[pid] = dict(item, id=pid, origin_bay=bay, source_side=side,
                                        target_side=target_side, direction=self._direction(side, target_side),
                                        task_type="cross" if cross else "same", status="stacked", batch_id=None)
                self.bay_stacks[bay].append(pid)
        expected_cross = round(self.rho * (self.N // 2))
        if left_count != self.K_s // 2 or cross_counts != [expected_cross, expected_cross]:
            raise ValueError(f"{path}: instance does not match the configured source split or cross-region ratio")

    def _available_batch_limits(self, bay):
        """Independent maxima for the two source-state features under Eqs. (3)-(8)."""
        stack = self.bay_stacks[bay]
        ref = self.plates[stack[0]]
        reserved = self._reserved_plate_ids()
        neighbours = [self.plates[item['plate_id']]['weight']
                      for item in self._neighbor_batch_candidates(bay, ref, self.active_crane)]
        best_size, best_mass, prefix_mass = 0, 0., 0.
        for size, pid in enumerate(stack[:self.B_max], 1):
            p = self.plates[pid]
            if (pid in reserved or p['status'] != 'stacked' or p['task_type'] != ref['task_type']
                    or abs(p['target_bay'] - ref['target_bay']) > self.epsilon_d):
                break
            prefix_mass += p['weight']
            if prefix_mass > self.W_max:
                break
            for extra in range(min(self.B_max-size, len(neighbours))+1):
                for weights in combinations(neighbours, extra):
                    mass = prefix_mass + sum(weights)
                    if mass <= self.W_max:
                        best_size, best_mass = max(best_size, size+extra), max(best_mass, mass)
        return best_size, best_mass

    def _decode_agent_action(self, action):
        if 0 <= action < 27:
            return dict(kind='origin', bay=SOURCE_BAYS[action], internal_action=SOURCE_BAYS[action])
        if 27 <= action < 33:
            slot = action-27
            return dict(kind='handshake', slot_id=slot, bay=self.handshake_slot_to_bay[slot],
                        internal_action=self.handshake_slot_to_action[slot])
        raise ValueError(f"Unknown action {action}")

    def action_masks(self):
        if self._crane_has_pending_operation(self.active_crane):
            return np.zeros(33, dtype=bool)
        return np.asarray([self._ordinary_bay_action_is_legal(b) for b in SOURCE_BAYS]
                          + [self._handshake_slot_action_is_legal(i) for i in range(6)], dtype=bool)

    def _pending(self):
        ops = [op for op in self.crane_tasks if op is not None]
        if self.remaininfor:
            ops.append(self.remaininfor['operation'])
        return ops

    def _used_slots(self):
        reserved = self._reserved_slot_actions()
        return sum(s['occupied'] or s['reserved'] or s['action'] in reserved for s in self.handshake_slots)

    def _has_available_handshake_slot(self):
        return self._used_slots() < self.capacity

    def _observe(self):
        mask = self.action_masks()
        src = np.zeros((27, 16), np.float32)
        slots = np.zeros((6, 8), np.float32)
        quality = np.zeros((33, 7), np.float32)
        valid = np.array([b in self.bay_stacks for b in SOURCE_BAYS], dtype=np.int8)
        pos = self.crane_positions[self.active_crane]
        used = self._used_slots()
        pending = self._pending()
        plans = {int(a): self._calculate_operation_plan(self.active_crane, int(a)) for a in np.flatnonzero(mask)}
        for i, bay in enumerate(SOURCE_BAYS):
            if not valid[i]:
                continue
            stack = self.bay_stacks[bay]
            src[i,:3] = (bay/41, int(bay in RIGHT_STORAGE_BAYS), len(stack)/20)
            for j, pid in enumerate(stack[:3]):
                p = self.plates[pid]
                src[i,3+3*j:6+3*j] = (p['target_bay']/41, p['weight']/20, p['task_type']=='cross')
            size, mass = self._available_batch_limits(bay) if mask[i] else (0, 0.)
            src[i,12:] = (mask[i], size / self.B_max, mass / self.W_max, abs(pos-bay)/41)
        for i, slot in enumerate(self.handshake_slots):
            incoming = next((op for op in pending if op['operation_type']=='cross_in'
                             and op['reserved_slot_action']==slot['action']), None)
            batch = self._batch_for_slot(slot) or (incoming['batch'] if incoming else None)
            targets = self._batch_target_bays(batch) if batch else []
            slots[i] = (slot['bay']/41, 1.0 if slot['occupied'] else 0.5 if incoming or slot['reserved'] else 0.,
                        np.mean(targets)/41 if targets else 0., min(targets)/41 if targets else 0.,
                        max(targets)/41 if targets else 0., mask[27+i], abs(pos-slot['bay'])/41, self.active_crane)
        for a, op in plans.items():
            overlap = sum(max(0., min(v['end'],w['end']) - max(v['start'],w['start']))
                          for p in pending if p['crane_id'] != self.active_crane
                          for v in self._mutex_windows(op) for w in self._mutex_windows(p))
            kind = op['operation_type']
            projected = used + int(kind=='cross_in') - int(kind=='cross_out')
            quality[a] = np.clip((op['empty_time']/164, op['loaded_time']/164,
                self._batch_size(op['batch'])/3, self._batch_weight(op['batch'])/20,
                (op['completion_time']-self.current_time)/182, overlap/182, projected/self.capacity), 0., 1.)
        return dict(source=src, handover=slots, quality=quality, source_mask=valid, action_mask=mask.astype(np.int8))

    def step(self, action):
        if not self.action_space.contains(action) or not self.action_masks()[action]:
            raise ValueError(f"Illegal candidate {action}")
        before = self.current_makespan
        self._dispatch(int(action))
        self.current_step += 1
        pending = self._pending()
        self.current_makespan = max([before, self.current_time] + [p['completion_time'] for p in pending])
        return (self._observe(), before-self.current_makespan, self._is_terminated(), False,
                {"makespan": self.current_makespan, "completed_plates": len(self.completed_plate_ids)})

    def _dispatch(self, action):
        active_crane = int(self.active_crane)
        t_now = self.current_time
        current_op = self._calculate_operation_plan(active_crane, action)
        committed_ops = []
        active_crane_selected = False

        if self.remaininfor is not None:
            remain_op = self.remaininfor["operation"]
            self._resolve_with_remain_operation(current_op, remain_op, t_now)
            if self._operation_completes_before(current_op, remain_op):
                committed_op = current_op
                self._commit_operation(committed_op)
                committed_ops.append(committed_op)
                self._sync_remaininfor_from_operation(self.remaininfor)
                active_crane_selected = self._advance_remain_or_select_after_faster_commit(
                    committed_op, committed_ops
                )
            else:
                committed_op = remain_op
                self._commit_operation(committed_op)
                committed_ops.append(committed_op)
                self.remaininfor = self._make_remaininfor(current_op)
                active_crane_selected = self._advance_remain_or_select_after_faster_commit(
                    committed_op, committed_ops
                )

        else:
            self.crane_tasks[active_crane] = current_op
            other_crane = 1 - active_crane
            if (
                self.crane_tasks[other_crane] is None
                and self._has_selectable_action_for_crane(other_crane)
            ):
                self.active_crane = other_crane
                return

            if self.crane_tasks[other_crane] is None:
                committed_op = current_op
                self._commit_operation(committed_op)
                committed_ops.append(committed_op)
                self.crane_tasks[active_crane] = None
            else:
                op_a = self.crane_tasks[active_crane]
                op_b = self.crane_tasks[other_crane]
                self._resolve_two_operations(op_a, op_b, t_now)
                if self._operation_completes_before(op_a, op_b):
                    committed_op = op_a
                    slow_op = op_b
                else:
                    committed_op = op_b
                    slow_op = op_a
                self._commit_operation(committed_op)
                committed_ops.append(committed_op)
                self.remaininfor = self._make_remaininfor(slow_op)
                self.crane_tasks = [None, None]
                active_crane_selected = self._advance_remain_or_select_after_faster_commit(
                    committed_op, committed_ops
                )

        self._auto_clear_if_needed()
        if not active_crane_selected:
            self._select_next_active_crane(committed_ops[-1] if committed_ops else None)

    @staticmethod
    def _operation_completes_before(op_a: dict[str, Any], op_b: dict[str, Any]) -> bool:
        if float(op_a["completion_time"]) != float(op_b["completion_time"]):
            return float(op_a["completion_time"]) < float(op_b["completion_time"])
        return int(op_a["crane_id"]) <= int(op_b["crane_id"])

    def _crane_has_pending_operation(self, crane_id: int) -> bool:
        crane_id = int(crane_id)
        if self.crane_tasks[crane_id] is not None:
            return True
        return (
            self.remaininfor is not None
            and int(self.remaininfor["crane_id"]) == crane_id
        )

    def _has_selectable_action_for_crane(self, crane_id: int) -> bool:
        crane_id = int(crane_id)
        if self._crane_has_pending_operation(crane_id):
            return False
        old_active = self.active_crane
        self.active_crane = crane_id
        try:
            return bool(np.any(self.action_masks()))
        finally:
            self.active_crane = old_active

    def _select_next_active_crane(self, committed_op: dict[str, Any] | None) -> None:
        preferred = int(committed_op["crane_id"]) if committed_op is not None else int(self.active_crane)
        other = 1 - preferred
        if self._has_selectable_action_for_crane(preferred):
            self.active_crane = preferred
        elif self._has_selectable_action_for_crane(other):
            self.active_crane = other
        else:
            self.active_crane = preferred

    def _is_terminated(self) -> bool:
        all_origin_stacks_empty = all(
            len(self.bay_stacks.get(int(bay), [])) == 0 for bay in self.origin_bays
        )
        all_handshake_slots_empty = all(
            not bool(slot.get("occupied", False)) and slot.get("batch_id") is None
            for slot in self.handshake_slots
        )
        return (
            all_origin_stacks_empty
            and all_handshake_slots_empty
            and self.remaininfor is None
            and self.crane_tasks == [None, None]
            and len(self.completed_plate_ids) == self.N
        )

    def _crane_is_busy(self, crane_id: int) -> bool:
        return self._crane_has_pending_operation(crane_id)

    def _auto_clear_if_needed(self) -> list[dict[str, Any]]:
        committed: list[dict[str, Any]] = []
        for crane_id in (LEFT_SIDE, RIGHT_SIDE):
            if self._crane_is_busy(crane_id):
                continue
            if not self._crane_inside_handshake(crane_id):
                continue
            if self._has_selectable_action_for_crane(crane_id):
                continue

            clear_op = self._make_clearance_only_operation(crane_id)
            if self.remaininfor is not None:
                self._resolve_with_remain_operation(
                    clear_op,
                    self.remaininfor["operation"],
                    self.current_time,
                )
            self._commit_operation(clear_op)
            committed.append(clear_op)
        return committed

    def _advance_remain_or_select_after_faster_commit(
        self,
        faster_op: dict[str, Any],
        committed_ops: list[dict[str, Any]],
    ) -> bool:
        faster_crane = int(faster_op["crane_id"])
        if self._has_selectable_action_for_crane(faster_crane):
            self.active_crane = faster_crane
            return True

        if self.remaininfor is None:
            return False

        slow_op = self.remaininfor["operation"]
        slow_crane = int(slow_op["crane_id"])
        self._commit_operation(slow_op)
        committed_ops.append(slow_op)
        self.remaininfor = None

        if self._has_selectable_action_for_crane(faster_crane):
            self.active_crane = faster_crane
        elif self._has_selectable_action_for_crane(slow_crane):
            self.active_crane = slow_crane
        elif self._is_terminated():
            self.active_crane = faster_crane
        else:
            raise RuntimeError(
                "no selectable action after auto-committing remain operation; "
                "action mask or state update is inconsistent"
            )
        return True

    def _commit_operation(self, operation: dict[str, Any]) -> None:
        operation_type = operation["operation_type"]
        if operation_type == "same":
            batch_id = self._register_operation_batch(operation, "completed")
            self._remove_operation_plates_from_stacks(operation)
            for plate_id in operation["plate_ids"]:
                plate = self.plates[int(plate_id)]
                plate["status"] = "completed"
                plate["batch_id"] = batch_id
                self.completed_plate_ids.add(int(plate_id))
        elif operation_type == "cross_in":
            batch_id = self._register_operation_batch(operation, "in_handshake")
            self._remove_operation_plates_from_stacks(operation)
            for plate_id in operation["plate_ids"]:
                plate = self.plates[int(plate_id)]
                plate["status"] = "in_handshake"
                plate["batch_id"] = batch_id
            slot = self._slot_by_action(int(operation["reserved_slot_action"]))
            slot["occupied"] = True
            slot["batch_id"] = batch_id
            slot["reserved"] = False
        elif operation_type == "cross_out":
            slot = self._slot_by_action(int(operation["action"]))
            batch_id = int(slot["batch_id"])
            batch = self.batches[batch_id]
            batch["status"] = "completed"
            for plate_id in operation["plate_ids"]:
                plate = self.plates[int(plate_id)]
                plate["status"] = "completed"
                self.completed_plate_ids.add(int(plate_id))
            slot["occupied"] = False
            slot["batch_id"] = None
            slot["reserved"] = False
        elif operation_type == "clearance_only":
            pass
        else:
            raise ValueError(f"unsupported operation_type {operation_type}")

        crane_id = int(operation["crane_id"])
        self.crane_positions[crane_id] = int(operation["end_bay"])
        self.crane_trajectories[crane_id].append(operation.copy())
        self.current_time = max(self.current_time, float(operation["completion_time"]))
        if self.remaininfor is not None and self.remaininfor.get("operation") is operation:
            self.remaininfor = None

    def _make_clearance_only_operation(self, crane_id: int) -> dict[str, Any]:
        crane_id = int(crane_id)
        current_bay = int(self.crane_positions[crane_id])
        safe_bay = self._safe_bay_for_crane(crane_id)
        empty_time = self._move_time(current_bay, safe_bay)
        clearance_window = {
            "type": "clearance",
            "start": float(self.current_time),
            "end": float(self.current_time + empty_time),
            "from_bay": current_bay,
            "safe_bay": safe_bay,
            "crane_id": crane_id,
        }
        return {
            "operation_id": f"clearance-{self.current_step}-{crane_id}",
            "operation_type": "clearance_only",
            "crane_id": crane_id,
            "action": None,
            "batch_id": None,
            "batch": {},
            "plate_ids": [],
            "pickup_sequence": [],
            "drop_sequence": [],
            "pickup_bays": [],
            "drop_bays": [],
            "empty_time": float(empty_time),
            "loaded_time": 0.0,
            "vertical_time": 0.0,
            "wait_time": 0.0,
            "handshake_window": None,
            "clearance_window": clearance_window,
            "completion_time": float(self.current_time + empty_time),
            "end_bay": safe_bay,
            "reserved_plate_ids": set(),
            "reserved_slot_action": None,
            "starts_inside_handshake": True,
            "clears_handshake": True,
            "has_waited": False,
        }

    def _register_operation_batch(self, operation: dict[str, Any], status: str) -> int:
        batch = operation.get("batch", {}).copy()
        batch_id = batch.get("batch_id", batch.get("id"))
        if isinstance(batch_id, str) or batch_id is None:
            batch_id = len(self.batches)
        batch_id = int(batch_id)
        while batch_id in self.batches:
            batch_id += 1
        batch["id"] = batch_id
        batch["batch_id"] = batch_id
        batch["status"] = status
        batch["plate_ids"] = [int(pid) for pid in operation["plate_ids"]]
        batch["pickup_sequence"] = [int(pid) for pid in operation["pickup_sequence"]]
        batch["drop_sequence"] = [int(pid) for pid in operation["drop_sequence"]]
        batch["target_bays"] = [int(self.plates[int(pid)]["target_bay"]) for pid in operation["plate_ids"]]
        batch["weight"] = float(sum(float(self.plates[int(pid)]["weight"]) for pid in operation["plate_ids"]))
        batch["batch_size"] = len(operation["plate_ids"])
        self.batches[batch_id] = batch
        operation["batch_id"] = batch_id
        return batch_id

    def _remove_operation_plates_from_stacks(self, operation: dict[str, Any]) -> None:
        batch = operation.get("batch", {})
        pickup_bays = batch.get("pickup_bays")
        if pickup_bays is None:
            pickup_bays = [self.plates[int(pid)]["origin_bay"] for pid in operation["pickup_sequence"]]
        for plate_id, bay in zip(operation["pickup_sequence"], pickup_bays):
            stack = self.bay_stacks[int(bay)]
            if not stack or int(stack[0]) != int(plate_id):
                raise RuntimeError(f"plate {plate_id} is not top of bay {bay} at commit")
            stack.pop(0)

    def _slot_by_action(self, action: int) -> dict[str, Any]:
        slot_id = self.handshake_action_to_slot[int(action)]
        return self.handshake_slots[slot_id]

    def _init_handshake_slots(self) -> None:
        self.handshake_slots = [
            {
                "slot": slot,
                "action": self.handshake_slot_to_action[slot],
                "bay": self.handshake_slot_to_bay[slot],
                "occupied": False,
                "batch_id": None,
                "reserved": False,
            }
            for slot in range(6)
        ]

    def _ordinary_bay_action_is_legal(self, action: int) -> bool:
        action = int(action)
        if action not in self.left_storage_bays and action not in self.right_storage_bays:
            return False
        if action not in self.origin_bays:
            return False
        stack = self.bay_stacks.get(action, [])
        if not stack:
            return False

        top_plate_id = int(stack[0])
        if top_plate_id in self._reserved_plate_ids():
            return False

        top_plate = self.plates[top_plate_id]
        if top_plate.get("status") != "stacked":
            return False
        if not self._plate_executable_by_current_crane(top_plate):
            return False
        if top_plate.get("task_type") == "cross" and not self._has_available_handshake_slot():
            return False
        return True

    def _plate_executable_by_current_crane(self, plate: dict[str, Any]) -> bool:
        return self._plate_executable_by_crane(plate, self.active_crane)

    def _plate_executable_by_crane(self, plate: dict[str, Any], crane_id: int) -> bool:
        current_side = int(crane_id)
        source_side = int(plate["source_side"])
        target_side = int(plate["target_side"])
        task_type = plate["task_type"]
        if task_type == "same":
            return source_side == current_side and target_side == current_side
        if task_type == "cross":
            return source_side == current_side and target_side != current_side
        return False

    def _build_batch_from_bay(self, action_bay: int, crane_id: int) -> dict[str, Any]:
        """Build a side-effect-free same-lift batch candidate from an origin bay."""
        action_bay = int(action_bay)
        crane_id = int(crane_id)
        if action_bay not in self.left_storage_bays and action_bay not in self.right_storage_bays:
            raise ValueError(f"action_bay {action_bay} is not a normal origin bay")
        if action_bay not in self.origin_bays:
            raise ValueError(f"action_bay {action_bay} is not an episode origin bay")

        stack = self.bay_stacks.get(action_bay, [])
        if not stack:
            raise ValueError(f"action_bay {action_bay} is empty")

        reference_plate_id = int(stack[0])
        reference = self.plates[reference_plate_id]
        if not self._batch_plate_can_start(reference, crane_id):
            raise ValueError(
                f"top plate {reference_plate_id} is not executable by crane {crane_id}"
            )

        pickup_sequence = [reference_plate_id]
        pickup_bays = [action_bay]
        batch_weight = float(reference["weight"])

        for plate_id in stack[1:3]:
            plate = self.plates[int(plate_id)]
            candidate_weight = batch_weight + float(plate["weight"])
            if not self._same_bay_lower_plate_can_join(
                plate=plate,
                reference=reference,
                candidate_size=len(pickup_sequence) + 1,
                candidate_weight=candidate_weight,
            ):
                break

            pickup_sequence.append(int(plate_id))
            pickup_bays.append(action_bay)
            batch_weight = candidate_weight

        if len(pickup_sequence) >= self.B_max or batch_weight >= self.W_max:
            return self._make_batch_dict(
                action_bay, crane_id, reference, pickup_sequence, pickup_bays
            )

        for candidate in self._neighbor_batch_candidates(action_bay, reference, crane_id):
            if len(pickup_sequence) + 1 > self.B_max:
                break

            plate_id = int(candidate["plate_id"])
            plate = self.plates[plate_id]
            candidate_weight = batch_weight + float(plate["weight"])
            if candidate_weight > self.W_max:
                continue

            pickup_sequence.append(plate_id)
            pickup_bays.append(int(candidate["bay"]))
            batch_weight = candidate_weight

            if len(pickup_sequence) >= self.B_max or batch_weight >= self.W_max:
                break

        return self._make_batch_dict(
            action_bay, crane_id, reference, pickup_sequence, pickup_bays
        )

    def _batch_plate_can_start(self, plate: dict[str, Any], crane_id: int) -> bool:
        if plate.get("status") != "stacked":
            return False
        if int(plate["id"]) in self._reserved_plate_ids():
            return False
        return self._plate_executable_by_crane(plate, crane_id)

    def _same_bay_lower_plate_can_join(
        self,
        plate: dict[str, Any],
        reference: dict[str, Any],
        candidate_size: int,
        candidate_weight: float,
    ) -> bool:
        if plate.get("status") != "stacked":
            return False
        if int(plate["id"]) in self._reserved_plate_ids():
            return False
        if plate["task_type"] != reference["task_type"]:
            return False
        if plate["direction"] != reference["direction"]:
            return False
        if abs(int(plate["target_bay"]) - int(reference["target_bay"])) > self.epsilon_d:
            return False
        if candidate_size > self.B_max:
            return False
        if candidate_weight > self.W_max:
            return False
        return True

    def _neighbor_batch_candidates(
        self,
        ref_bay: int,
        reference: dict[str, Any],
        crane_id: int,
    ) -> list[dict[str, int]]:
        candidate_bays = [
            bay
            for bay in self.origin_bays
            if bay != ref_bay and 0 < abs(int(bay) - int(ref_bay)) <= self.epsilon_o
        ]
        candidates = []
        for bay in candidate_bays:
            stack = self.bay_stacks.get(int(bay), [])
            if not stack:
                continue

            plate_id = int(stack[0])
            plate = self.plates[plate_id]
            if not self._neighbor_plate_can_join(plate, reference, crane_id):
                continue

            candidates.append(
                {
                    "bay": int(bay),
                    "plate_id": plate_id,
                    "origin_distance": abs(int(bay) - int(ref_bay)),
                    "target_distance": abs(
                        int(plate["target_bay"]) - int(reference["target_bay"])
                    ),
                    "height": len(stack),
                }
            )

        candidates.sort(key=lambda item: self._neighbor_candidate_sort_key(item, reference))
        return candidates

    def _neighbor_plate_can_join(
        self,
        plate: dict[str, Any],
        reference: dict[str, Any],
        crane_id: int,
    ) -> bool:
        if plate.get("status") != "stacked":
            return False
        if int(plate["id"]) in self._reserved_plate_ids():
            return False
        if not self._plate_executable_by_crane(plate, crane_id):
            return False
        if plate["task_type"] != reference["task_type"]:
            return False
        if plate["direction"] != reference["direction"]:
            return False
        if abs(int(plate["target_bay"]) - int(reference["target_bay"])) > self.epsilon_d:
            return False
        return True

    def _neighbor_candidate_sort_key(
        self,
        item: dict[str, int],
        reference: dict[str, Any],
    ) -> tuple[int, int, int, int, int, int]:
        bay = int(item["bay"])
        ref_bay = int(reference["origin_bay"])
        direction = reference["direction"]
        if direction == "L2R":
            direction_preference = 0 if bay > ref_bay else 1
        elif direction == "R2L":
            direction_preference = 0 if bay < ref_bay else 1
        else:
            direction_preference = 0

        same_target_priority = 0 if int(item["target_distance"]) == 0 else 1
        return (
            same_target_priority,
            int(item["origin_distance"]),
            direction_preference,
            int(item["target_distance"]),
            -int(item["height"]),
            bay,
        )

    def _make_batch_dict(
        self,
        action_bay: int,
        crane_id: int,
        reference: dict[str, Any],
        pickup_sequence: list[int],
        pickup_bays: list[int],
    ) -> dict[str, Any]:
        target_bays = [
            int(self.plates[int(plate_id)]["target_bay"]) for plate_id in pickup_sequence
        ]
        batch_weight = sum(
            float(self.plates[int(plate_id)]["weight"]) for plate_id in pickup_sequence
        )
        return {
            "origin_bay": int(action_bay),
            "crane_id": int(crane_id),
            "reference_plate_id": int(reference["id"]),
            "plate_ids": list(pickup_sequence),
            "pickup_sequence": list(pickup_sequence),
            "pickup_bays": list(pickup_bays),
            "drop_sequence": list(reversed(pickup_sequence)),
            "batch_size": len(pickup_sequence),
            "batch_weight": float(batch_weight),
            "weight": float(batch_weight),
            "target_bays": target_bays,
            "target_bay_mean": float(np.mean(target_bays)),
            "target_bay_min": int(min(target_bays)),
            "target_bay_max": int(max(target_bays)),
            "task_type": reference["task_type"],
            "direction": reference["direction"],
            "source_side": int(reference["source_side"]),
            "target_side": int(reference["target_side"]),
            "status": "candidate",
        }

    def _calculate_operation_plan(self, crane_id: int, action: int) -> dict[str, Any]:
        crane_id = int(crane_id)
        action = int(action)
        if crane_id not in (LEFT_SIDE, RIGHT_SIDE):
            raise ValueError(f"invalid crane_id {crane_id}")
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"action {action} outside Discrete({self.action_space.n})")

        decoded = self._decode_agent_action(action)
        internal_action = int(decoded["internal_action"])

        if decoded["kind"] == "handshake":
            return self._calculate_cross_out_plan(crane_id, internal_action)

        if decoded["kind"] == "origin":
            bay = int(decoded["bay"])
            batch = self._build_batch_from_bay(bay, crane_id)
            if batch["task_type"] == "same":
                return self._calculate_same_plan(crane_id, bay, batch)
            if batch["task_type"] == "cross":
                return self._calculate_cross_in_plan(crane_id, bay, batch)
            raise ValueError(f"unsupported task_type {batch['task_type']}")

        raise ValueError(f"unsupported decoded action kind {decoded['kind']}")

    def _calculate_same_plan(
        self,
        crane_id: int,
        action: int,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        planning_start_bay, clearance_window, clearance_time = (
            self._planning_start_after_clearance(crane_id)
        )
        pickup_bays = self._compress_consecutive_bays(batch["pickup_bays"])
        drop_sequence = self._batch_drop_sequence(batch)
        drop_bays = self._compress_consecutive_bays(
            [int(self.plates[plate_id]["target_bay"]) for plate_id in drop_sequence]
        )
        timing = self._timing_pickup_then_drop(
            current_bay=planning_start_bay,
            pickup_bays=pickup_bays,
            drop_bays=drop_bays,
        )
        if clearance_window is not None:
            timing["empty_time"] += clearance_time
            timing["completion_time"] += clearance_time
        return self._make_operation_plan(
            operation_type="same",
            crane_id=crane_id,
            action=action,
            batch=batch,
            pickup_bays=pickup_bays,
            drop_bays=drop_bays,
            timing=timing,
            handshake_window=None,
            clearance_window=clearance_window,
            reserved_slot_action=None,
            starts_inside_handshake=clearance_window is not None,
            clears_handshake=clearance_window is not None,
        )

    def _calculate_cross_in_plan(
        self,
        crane_id: int,
        action: int,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        slot = self._select_handshake_slot_for_cross_in(batch)
        slot_bay = int(slot["bay"])
        planning_start_bay, clearance_window, clearance_time = (
            self._planning_start_after_clearance(crane_id)
        )
        pickup_bays = self._compress_consecutive_bays(batch["pickup_bays"])
        timing = self._timing_pickup_then_slot(
            current_bay=planning_start_bay,
            pickup_bays=pickup_bays,
            slot_bay=slot_bay,
            crane_id=crane_id,
        )
        if clearance_window is not None:
            timing["empty_time"] += clearance_time
            timing["completion_time"] += clearance_time
            timing["slot_vertical_start"] += clearance_time
            timing["slot_vertical_end"] += clearance_time
            timing["handshake_start"] += clearance_time
            timing["handshake_end"] += clearance_time
        handshake_window = {
            "type": "in",
            "slot": int(slot["slot"]),
            "slot_action": int(slot["action"]),
            "bay": slot_bay,
            "safe_bay": self._safe_bay_for_crane(crane_id),
            "start": float(timing["handshake_start"]),
            "end": float(timing["handshake_end"]),
            "crane_id": crane_id,
        }
        if (
            clearance_window is not None
            and float(clearance_window["end"]) > float(handshake_window["start"])
        ):
            raise RuntimeError("cross_in clearance_window overlaps handshake_window")
        return self._make_operation_plan(
            operation_type="cross_in",
            crane_id=crane_id,
            action=action,
            batch=batch,
            pickup_bays=pickup_bays,
            drop_bays=[slot_bay],
            timing=timing,
            handshake_window=handshake_window,
            clearance_window=clearance_window,
            reserved_slot_action=int(slot["action"]),
            starts_inside_handshake=clearance_window is not None,
            clears_handshake=clearance_window is not None,
        )

    def _calculate_cross_out_plan(self, crane_id: int, action: int) -> dict[str, Any]:
        slot_id = self.handshake_action_to_slot[int(action)]
        slot = self.handshake_slots[slot_id]
        if not self._handshake_slot_action_is_legal_for_crane(slot_id, crane_id):
            raise ValueError(f"handshake action {action} is not legal for crane {crane_id}")

        batch = self._batch_for_slot(slot)
        if batch is None:
            raise ValueError(f"slot {slot_id} does not contain a batch")

        slot_bay = int(slot["bay"])
        drop_sequence = self._batch_drop_sequence(batch)
        drop_bays = self._compress_consecutive_bays(
            [int(self.plates[plate_id]["target_bay"]) for plate_id in drop_sequence]
        )
        timing = self._timing_slot_then_drop(
            current_bay=int(self.crane_positions[crane_id]),
            slot_bay=slot_bay,
            drop_bays=drop_bays,
            crane_id=crane_id,
        )
        starts_inside = int(self.crane_positions[crane_id]) in self.handshake_bays
        safe_bay = self._safe_bay_for_crane(crane_id)
        handshake_window = None
        clearance_window = None
        clears_handshake = False
        if starts_inside:
            clearance_window = {
                "type": "clearance",
                "start": float(timing["clearance_start"]),
                "end": float(timing["clearance_end"]),
                "slot": slot_id,
                "slot_action": int(action),
                "bay": slot_bay,
                "slot_bay": slot_bay,
                "safe_bay": safe_bay,
                "crane_id": crane_id,
            }
            clears_handshake = True
        else:
            handshake_window = {
                "type": "out",
                "slot": slot_id,
                "slot_action": int(action),
                "bay": slot_bay,
                "safe_bay": safe_bay,
                "start": float(timing["handshake_start"]),
                "end": float(timing["handshake_end"]),
                "crane_id": crane_id,
            }

        return self._make_operation_plan(
            operation_type="cross_out",
            crane_id=crane_id,
            action=action,
            batch=batch,
            pickup_bays=[slot_bay],
            drop_bays=drop_bays,
            timing=timing,
            handshake_window=handshake_window,
            clearance_window=clearance_window,
            reserved_slot_action=int(action),
            starts_inside_handshake=starts_inside,
            clears_handshake=clears_handshake,
        )

    def _make_operation_plan(
        self,
        operation_type: str,
        crane_id: int,
        action: int,
        batch: dict[str, Any],
        pickup_bays: list[int],
        drop_bays: list[int],
        timing: dict[str, float],
        handshake_window: dict[str, Any] | None,
        clearance_window: dict[str, Any] | None,
        reserved_slot_action: int | None,
        starts_inside_handshake: bool,
        clears_handshake: bool,
    ) -> dict[str, Any]:
        pickup_sequence = self._batch_plate_ids(batch)
        drop_sequence = self._batch_drop_sequence(batch)
        batch_id = batch.get("batch_id", batch.get("id"))
        if batch_id is None:
            batch_id = f"candidate-{self.current_step}-{crane_id}-{action}"
        operation_id = f"op-{self.current_step}-{crane_id}-{action}"
        return {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "crane_id": int(crane_id),
            "action": int(action),
            "batch_id": batch_id,
            "batch": batch.copy(),
            "plate_ids": list(pickup_sequence),
            "pickup_sequence": list(pickup_sequence),
            "drop_sequence": list(drop_sequence),
            "pickup_bays": list(pickup_bays),
            "drop_bays": list(drop_bays),
            "empty_time": float(timing["empty_time"]),
            "loaded_time": float(timing["loaded_time"]),
            "vertical_time": float(timing["vertical_time"]),
            "wait_time": 0.0,
            "handshake_window": handshake_window,
            "clearance_window": clearance_window,
            "completion_time": float(timing["completion_time"]),
            "end_bay": int(timing["end_bay"]),
            "reserved_plate_ids": set(int(pid) for pid in pickup_sequence),
            "reserved_slot_action": reserved_slot_action,
            "starts_inside_handshake": bool(starts_inside_handshake),
            "clears_handshake": bool(clears_handshake),
            "has_waited": False,
        }

    def _timing_pickup_then_drop(
        self,
        current_bay: int,
        pickup_bays: list[int],
        drop_bays: list[int],
    ) -> dict[str, float]:
        if not pickup_bays or not drop_bays:
            raise ValueError("pickup_bays and drop_bays must be non-empty")
        t = float(self.current_time)
        empty_time = self._move_time(current_bay, pickup_bays[0])
        t += empty_time
        loaded_time = 0.0
        vertical_time = 0.0

        for i, bay in enumerate(pickup_bays):
            t += self.vertical_cycle
            vertical_time += self.vertical_cycle
            if i < len(pickup_bays) - 1:
                move = self._move_time(bay, pickup_bays[i + 1])
                loaded_time += move
                t += move

        move = self._move_time(pickup_bays[-1], drop_bays[0])
        loaded_time += move
        t += move

        for i, bay in enumerate(drop_bays):
            t += self.vertical_cycle
            vertical_time += self.vertical_cycle
            if i < len(drop_bays) - 1:
                move = self._move_time(bay, drop_bays[i + 1])
                loaded_time += move
                t += move

        return {
            "empty_time": float(empty_time),
            "loaded_time": float(loaded_time),
            "vertical_time": float(vertical_time),
            "completion_time": float(t),
            "end_bay": float(drop_bays[-1]),
        }

    def _timing_pickup_then_slot(
        self,
        current_bay: int,
        pickup_bays: list[int],
        slot_bay: int,
        crane_id: int,
    ) -> dict[str, float]:
        if not pickup_bays:
            raise ValueError("pickup_bays must be non-empty")
        safe_bay = self._safe_bay_for_crane(crane_id)
        t = float(self.current_time)
        empty_time = self._move_time(current_bay, pickup_bays[0])
        t += empty_time
        loaded_time = 0.0
        vertical_time = 0.0

        for i, bay in enumerate(pickup_bays):
            t += self.vertical_cycle
            vertical_time += self.vertical_cycle
            if i < len(pickup_bays) - 1:
                move = self._move_time(bay, pickup_bays[i + 1])
                loaded_time += move
                t += move

        move = self._move_time(pickup_bays[-1], safe_bay)
        loaded_time += move
        t += move
        handshake_start = t

        move = self._move_time(safe_bay, slot_bay)
        loaded_time += move
        t += move
        slot_vertical_start = t
        t += self.vertical_cycle
        vertical_time += self.vertical_cycle
        slot_vertical_end = t
        move = self._move_time(slot_bay, safe_bay)
        empty_time += move
        t += move
        handshake_end = t

        return {
            "empty_time": float(empty_time),
            "loaded_time": float(loaded_time),
            "vertical_time": float(vertical_time),
            "slot_vertical_start": float(slot_vertical_start),
            "slot_vertical_end": float(slot_vertical_end),
            "handshake_start": float(handshake_start),
            "handshake_end": float(handshake_end),
            "completion_time": float(t),
            "end_bay": float(safe_bay),
        }

    def _timing_slot_then_drop(
        self,
        current_bay: int,
        slot_bay: int,
        drop_bays: list[int],
        crane_id: int,
    ) -> dict[str, float]:
        if not drop_bays:
            raise ValueError("drop_bays must be non-empty")
        safe_bay = self._safe_bay_for_crane(crane_id)
        t = float(self.current_time)
        empty_time = 0.0
        loaded_time = 0.0

        if int(current_bay) in self.handshake_bays:
            clearance_start = t
            move = self._move_time(current_bay, slot_bay)
            empty_time += move
            t += move
        else:
            move = self._move_time(current_bay, safe_bay)
            empty_time += move
            t += move
            handshake_start = t
            move = self._move_time(safe_bay, slot_bay)
            empty_time += move
            t += move

        slot_vertical_start = t
        t += self.vertical_cycle
        vertical_time = self.vertical_cycle
        slot_vertical_end = t

        move = self._move_time(slot_bay, safe_bay)
        loaded_time += move
        t += move
        if int(current_bay) in self.handshake_bays:
            clearance_end = t
            handshake_start = None
            handshake_end = None
        else:
            clearance_start = None
            clearance_end = None
            handshake_end = t

        move = self._move_time(safe_bay, drop_bays[0])
        loaded_time += move
        t += move

        for i, bay in enumerate(drop_bays):
            t += self.vertical_cycle
            vertical_time += self.vertical_cycle
            if i < len(drop_bays) - 1:
                move = self._move_time(bay, drop_bays[i + 1])
                loaded_time += move
                t += move

        return {
            "empty_time": float(empty_time),
            "loaded_time": float(loaded_time),
            "vertical_time": float(vertical_time),
            "slot_vertical_start": float(slot_vertical_start),
            "slot_vertical_end": float(slot_vertical_end),
            "handshake_start": None if handshake_start is None else float(handshake_start),
            "handshake_end": None if handshake_end is None else float(handshake_end),
            "clearance_start": None if clearance_start is None else float(clearance_start),
            "clearance_end": None if clearance_end is None else float(clearance_end),
            "completion_time": float(t),
            "end_bay": float(drop_bays[-1]),
        }

    def _select_handshake_slot_for_cross_in(self, batch: dict[str, Any]) -> dict[str, Any]:
        direction = batch.get("direction")
        if direction == "L2R":
            action_order = [19, 42, 20, 43, 21, 44]
        elif direction == "R2L":
            action_order = [21, 44, 20, 43, 19, 42]
        else:
            raise ValueError(f"invalid cross_in batch direction {direction}")

        reserved_actions = self._reserved_slot_actions()
        for action in action_order:
            slot_id = self.handshake_action_to_slot[action]
            slot = self.handshake_slots[slot_id]
            if bool(slot.get("occupied", False)):
                continue
            if bool(slot.get("reserved", False)):
                continue
            if int(action) in reserved_actions:
                continue
            return slot
        raise ValueError("no available handshake slot for cross_in operation")

    def _handshake_slot_action_is_legal_for_crane(self, slot_id: int, crane_id: int) -> bool:
        old_active = self.active_crane
        self.active_crane = int(crane_id)
        try:
            return self._handshake_slot_action_is_legal(slot_id)
        finally:
            self.active_crane = old_active

    def _batch_plate_ids(self, batch: dict[str, Any]) -> list[int]:
        if "plate_ids" in batch:
            return [int(plate_id) for plate_id in batch["plate_ids"]]
        if "pickup_sequence" in batch:
            return [int(plate_id) for plate_id in batch["pickup_sequence"]]
        if "drop_sequence" in batch:
            return [int(plate_id) for plate_id in batch["drop_sequence"]]
        raise ValueError("batch has no plate id sequence")

    def _batch_drop_sequence(self, batch: dict[str, Any]) -> list[int]:
        if "drop_sequence" in batch:
            return [int(plate_id) for plate_id in batch["drop_sequence"]]
        return list(reversed(self._batch_plate_ids(batch)))

    def _move_time(self, start_bay: float, end_bay: float) -> float:
        return float(abs(float(end_bay) - float(start_bay)) * self.move_time_per_bay)

    @staticmethod
    def _compress_consecutive_bays(seq: list[int]) -> list[int]:
        compressed = []
        previous = None
        for bay in seq:
            bay = int(bay)
            if previous is None or bay != previous:
                compressed.append(bay)
            previous = bay
        return compressed

    def _make_remaininfor(self, operation: dict[str, Any]) -> dict[str, Any]:
        return {
            "crane_id": int(operation["crane_id"]),
            "operation": operation,
            "operation_id": operation["operation_id"],
            "batch_id": operation["batch_id"],
            "completion_time": float(operation["completion_time"]),
            "wait_time": float(operation["wait_time"]),
            "handshake_window": operation.get("handshake_window"),
            "clearance_window": operation.get("clearance_window"),
            "reserved_plate_ids": set(int(pid) for pid in operation.get("reserved_plate_ids", set())),
            "reserved_slot_action": operation.get("reserved_slot_action"),
            "end_bay": int(operation["end_bay"]),
            "has_waited": bool(operation.get("has_waited", False)),
        }

    def _sync_remaininfor_from_operation(self, remain: dict[str, Any]) -> None:
        operation = self._operation_from_remain(remain)
        if operation is remain:
            return
        remain["crane_id"] = int(operation["crane_id"])
        remain["operation_id"] = operation["operation_id"]
        remain["batch_id"] = operation["batch_id"]
        remain["completion_time"] = float(operation["completion_time"])
        remain["wait_time"] = float(operation["wait_time"])
        remain["handshake_window"] = operation.get("handshake_window")
        remain["clearance_window"] = operation.get("clearance_window")
        remain["reserved_plate_ids"] = set(int(pid) for pid in operation.get("reserved_plate_ids", set()))
        remain["reserved_slot_action"] = operation.get("reserved_slot_action")
        remain["end_bay"] = int(operation["end_bay"])
        remain["has_waited"] = bool(operation.get("has_waited", False))

    @staticmethod
    def _operation_from_remain(remain_or_operation: dict[str, Any]) -> dict[str, Any]:
        operation = remain_or_operation.get("operation")
        if isinstance(operation, dict):
            return operation
        return remain_or_operation

    def _shift_operation_time(
        self,
        operation: dict[str, Any],
        extra_wait: float,
        anchor_window: dict[str, Any] | None = None,
    ) -> None:
        extra_wait = float(extra_wait)
        if extra_wait < 0.0:
            raise ValueError("extra_wait must be non-negative")
        if extra_wait == 0.0:
            return

        operation["wait_time"] = float(operation.get("wait_time", 0.0)) + extra_wait
        operation["completion_time"] = float(operation["completion_time"]) + extra_wait
        operation["has_waited"] = True

        clearance_window = operation.get("clearance_window")
        handshake_window = operation.get("handshake_window")

        def _shift_window(window: dict[str, Any] | None) -> None:
            if window is None:
                return
            window["start"] = float(window["start"]) + extra_wait
            window["end"] = float(window["end"]) + extra_wait

        if anchor_window is None:
            _shift_window(clearance_window)
            _shift_window(handshake_window)
            return

        has_clearance = clearance_window is not None
        has_handshake = handshake_window is not None
        if not (has_clearance and has_handshake):
            _shift_window(clearance_window)
            _shift_window(handshake_window)
            return

        if anchor_window is clearance_window or anchor_window == clearance_window:
            _shift_window(clearance_window)
            _shift_window(handshake_window)
            return

        if anchor_window is handshake_window or anchor_window == handshake_window:
            if float(clearance_window["end"]) <= float(handshake_window["start"]):
                _shift_window(handshake_window)
                return
            _shift_window(clearance_window)
            _shift_window(handshake_window)
            return

        _shift_window(clearance_window)
        _shift_window(handshake_window)

    def _resolve_two_operations(
        self,
        op_a: dict[str, Any],
        op_b: dict[str, Any],
        t_now: float,
    ) -> None:
        del t_now
        self._resolve_window_conflicts(
            op_a,
            op_b,
            current_clearance_preempts=False,
            only_current_clearance=False,
        )

    def _resolve_with_remain_operation(
        self,
        current_op: dict[str, Any],
        remain_op: dict[str, Any],
        t_now: float,
    ) -> None:
        del t_now
        remain_operation = self._operation_from_remain(remain_op)

        changed_1 = False
        if current_op.get("clearance_window") is not None:
            changed_1 = self._resolve_window_conflicts(
                current_op,
                remain_operation,
                current_clearance_preempts=True,
                only_current_clearance=True,
            )

        changed_2 = self._resolve_window_conflicts(
            current_op,
            remain_operation,
            current_clearance_preempts=False,
            only_current_clearance=False,
        )

        if (
            (changed_1 or changed_2)
            and isinstance(self.remaininfor, dict)
            and self.remaininfor.get("operation") is remain_operation
        ):
            self._sync_remaininfor_from_operation(self.remaininfor)

    @staticmethod
    def _windows_overlap(win_a: dict[str, Any], win_b: dict[str, Any]) -> bool:
        return float(win_a["start"]) < float(win_b["end"]) and float(win_b["start"]) < float(win_a["end"])

    def _mutex_windows(self, operation: dict[str, Any]) -> list[dict[str, Any]]:
        windows = []
        clearance_window = operation.get("clearance_window")
        handshake_window = operation.get("handshake_window")

        for window in (clearance_window, handshake_window):
            if window is None:
                continue
            missing = [
                field
                for field in ("type", "start", "end", "crane_id")
                if field not in window
            ]
            if missing:
                raise RuntimeError(
                    f"mutex window missing fields {missing} in operation "
                    f"{operation.get('operation_id')}"
                )
            start = float(window["start"])
            end = float(window["end"])
            if end < start:
                raise RuntimeError(
                    "mutex window end is before start in operation "
                    f"{operation.get('operation_id')}"
                )
            windows.append(window)

        if clearance_window is not None and handshake_window is not None:
            if float(clearance_window["end"]) > float(handshake_window["start"]):
                raise RuntimeError(
                    "clearance_window overlaps handshake_window in operation "
                    f"{operation.get('operation_id')}"
                )

        return sorted(windows, key=lambda window: float(window["start"]))

    def _window_priority(
        self,
        window: dict[str, Any],
        operation: dict[str, Any],
    ) -> tuple[int, int]:
        type_order = {
            "clearance": 0,
            "out": 1,
            "in": 2,
        }
        type_priority = type_order.get(str(window.get("type")), 3)
        crane_id = int(window.get("crane_id", operation.get("crane_id", RIGHT_SIDE)))
        return type_priority, crane_id

    def _choose_window_winner(
        self,
        op_a: dict[str, Any],
        win_a: dict[str, Any],
        op_b: dict[str, Any],
        win_b: dict[str, Any],
        current_clearance_preempts: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if (
            current_clearance_preempts
            and str(win_a.get("type")) == "clearance"
        ):
            return op_a, win_a, op_b, win_b

        start_a = float(win_a["start"])
        start_b = float(win_b["start"])
        if start_a < start_b:
            return op_a, win_a, op_b, win_b
        if start_b < start_a:
            return op_b, win_b, op_a, win_a

        priority_a = self._window_priority(win_a, op_a)
        priority_b = self._window_priority(win_b, op_b)
        if priority_a <= priority_b:
            return op_a, win_a, op_b, win_b
        return op_b, win_b, op_a, win_a

    def _resolve_window_conflicts(
        self,
        op_a: dict[str, Any],
        op_b: dict[str, Any],
        *,
        current_clearance_preempts: bool = False,
        only_current_clearance: bool = False,
        max_iter: int = 30,
    ) -> bool:
        changed = False
        for _ in range(max_iter):
            windows_a = self._mutex_windows(op_a)
            windows_b = self._mutex_windows(op_b)
            if only_current_clearance:
                windows_a = [
                    window
                    for window in windows_a
                    if str(window.get("type")) == "clearance"
                ]

            found_overlap = False
            for win_a in windows_a:
                for win_b in windows_b:
                    if not self._windows_overlap(win_a, win_b):
                        continue

                    found_overlap = True
                    winner_op, winner_win, loser_op, loser_win = (
                        self._choose_window_winner(
                            op_a,
                            win_a,
                            op_b,
                            win_b,
                            current_clearance_preempts=current_clearance_preempts,
                        )
                    )
                    del winner_op
                    extra_wait = max(
                        0.0,
                        float(winner_win["end"]) - float(loser_win["start"]),
                    )
                    if extra_wait <= 0.0:
                        continue

                    self._shift_operation_time(
                        loser_op,
                        extra_wait,
                        anchor_window=loser_win,
                    )
                    changed = True
                    break

                if found_overlap:
                    break

            if not found_overlap:
                return changed

        raise RuntimeError("window conflict resolution did not converge")

    def _handshake_slot_action_is_legal(self, slot_id: int) -> bool:
        slot = self.handshake_slots[int(slot_id)]
        if not bool(slot.get("occupied", False)):
            return False
        if bool(slot.get("reserved", False)):
            return False
        if int(slot["action"]) in self._reserved_slot_actions():
            return False

        batch = self._batch_for_slot(slot)
        if batch is None:
            return False
        if batch.get("status") != "in_handshake":
            return False
        return int(batch.get("target_side", -1)) == int(self.active_crane)

    def _batch_for_slot(self, slot: dict[str, Any]) -> dict[str, Any] | None:
        batch_id = slot.get("batch_id")
        if batch_id is None:
            return None
        return self.batches.get(int(batch_id))

    def _batch_size(self, batch: dict[str, Any]) -> int:
        if "batch_size" in batch:
            return int(batch["batch_size"])
        if "plate_ids" in batch:
            return len(batch["plate_ids"])
        if "pickup_sequence" in batch:
            return len(batch["pickup_sequence"])
        return 1

    def _batch_weight(self, batch: dict[str, Any]) -> float:
        if "weight" in batch:
            return float(batch["weight"])
        if "total_weight" in batch:
            return float(batch["total_weight"])
        plate_ids = batch.get("plate_ids", batch.get("pickup_sequence", []))
        return float(sum(float(self.plates[int(pid)]["weight"]) for pid in plate_ids))

    def _batch_target_bays(self, batch: dict[str, Any]) -> list[int]:
        if "target_bays" in batch:
            return [int(bay) for bay in batch["target_bays"]]
        if "target_bay" in batch:
            return [int(batch["target_bay"])]
        plate_ids = batch.get("plate_ids", batch.get("pickup_sequence", []))
        return [
            int(self.plates[int(plate_id)]["target_bay"])
            for plate_id in plate_ids
            if int(plate_id) in self.plates
        ]

    def _reserved_plate_ids(self) -> set[int]:
        reserved = set(int(pid) for pid in self.reserved_plate_ids)
        for operation in self.crane_tasks:
            if isinstance(operation, dict):
                for pid in operation.get("reserved_plate_ids", []):
                    reserved.add(int(pid))
        if isinstance(self.remaininfor, dict):
            for pid in self.remaininfor.get("reserved_plate_ids", []):
                reserved.add(int(pid))
            operation = self.remaininfor.get("operation")
            if isinstance(operation, dict):
                for pid in operation.get("reserved_plate_ids", []):
                    reserved.add(int(pid))
        return reserved

    def _reserved_slot_actions(self) -> set[int]:
        reserved = set(int(action) for action in self.reserved_slot_actions)
        for operation in self.crane_tasks:
            if isinstance(operation, dict):
                action = operation.get("reserved_slot_action")
                if action is not None:
                    reserved.add(int(action))
        if isinstance(self.remaininfor, dict):
            action = self.remaininfor.get("reserved_slot_action")
            if action is not None:
                reserved.add(int(action))
            operation = self.remaininfor.get("operation")
            if isinstance(operation, dict):
                action = operation.get("reserved_slot_action")
                if action is not None:
                    reserved.add(int(action))
        return reserved

    def _crane_inside_handshake(self, crane_id: int) -> bool:
        return int(self.crane_positions[int(crane_id)]) in self.handshake_bays

    @staticmethod
    def _safe_bay_for_crane(crane_id: int) -> int:
        return LEFT_SAFE_BAY if int(crane_id) == LEFT_SIDE else RIGHT_SAFE_BAY

    def _planning_start_after_clearance(
        self,
        crane_id: int,
    ) -> tuple[int, dict[str, Any] | None, float]:
        crane_id = int(crane_id)
        current_bay = int(self.crane_positions[crane_id])
        if current_bay not in self.handshake_bays:
            return current_bay, None, 0.0

        safe_bay = self._safe_bay_for_crane(crane_id)
        clearance_time = self._move_time(current_bay, safe_bay)
        clearance_window = {
            "type": "clearance",
            "start": float(self.current_time),
            "end": float(self.current_time + clearance_time),
            "from_bay": current_bay,
            "safe_bay": safe_bay,
            "crane_id": crane_id,
        }
        return safe_bay, clearance_window, float(clearance_time)

    @staticmethod
    def _direction(source_side: int, target_side: int) -> str:
        if source_side == target_side:
            return "same"
        if source_side == LEFT_SIDE and target_side == RIGHT_SIDE:
            return "L2R"
        return "R2L"
