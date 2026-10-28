"""Original batch simulation adapter: fixed runtimes and CPU/GPU reservations.

The resource accounting below is a small original implementation. It does not
vendor Alibaba's simulator. Requests and capacities use integer hundredths of
a CPU core / GPU card; memory and GPU model compatibility are not simulated.
"""
from __future__ import annotations

import copy
import heapq

import numpy as np

DENOMINATOR = 300_000_000
WAIT_UNITS_PER_SECOND = 50_000
HIDDEN_FIELDS = {"patience_seconds", "deadline", "conditional_on_survived_to_score_start",
                 "cancel_time", "patience", "patience_deadline"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pricing(job):
    cpu = 60 * job["cpu_units"] * job["duration"]
    gpu = 2016 * job["gpu_units"] * job["duration"]
    total = cpu + gpu + 45
    return dict(cpu_value=cpu / DENOMINATOR, gpu_value=gpu / DENOMINATOR,
                resource_contract_value=(cpu + gpu) / DENOMINATOR,
                invocation_fee=45 / DENOMINATOR, contract_units=total,
                contract_value=total / DENOMINATOR, total_charge=total / DENOMINATOR)


class ResourceNode:
    """Exact reservation/release ledger with the experiment's Node interface."""
    def __init__(self, machine):
        self.num_cpus = self.idl_cpus = int(machine["cpu_units"])
        self.num_gpus = self.idl_gpus = int(machine["gpu_units"])
        self.job_cpus = self.job_gpus = 0
        self.job_runn_list = []

    def alloc_job(self, job):
        cpu, gpu = job["num_cpu"], job["num_gpu"]
        if cpu < 0 or gpu < 0 or cpu > self.idl_cpus or gpu > self.idl_gpus:
            return False
        self.idl_cpus -= cpu
        self.idl_gpus -= gpu
        self.job_cpus += cpu
        self.job_gpus += gpu
        self.job_runn_list.append(job)
        return True

    def release_job(self, job):
        if job not in self.job_runn_list:
            return False
        self.job_runn_list.remove(job)
        self.idl_cpus += job["num_cpu"]
        self.idl_gpus += job["num_gpu"]
        self.job_cpus -= job["num_cpu"]
        self.job_gpus -= job["num_gpu"]
        return True

    def check_rsrc(self):
        require(0 <= self.idl_cpus <= self.num_cpus and 0 <= self.idl_gpus <= self.num_gpus,
                "resource reservation outside capacity")
        require(self.num_cpus == self.idl_cpus + self.job_cpus
                and self.num_gpus == self.idl_gpus + self.job_gpus,
                "resource accounting identity failed")


class BatchEnvironment:
    def __init__(self, machines, jobs):
        self.machines, self.jobs = machines, jobs
        self.nodes = [ResourceNode(machine) for machine in machines]
        self.cap_cpu = np.array([node.num_cpus for node in self.nodes], dtype=np.int64)
        self.cap_gpu = np.array([node.num_gpus for node in self.nodes], dtype=np.int64)
        self.pending, self.events, self.assignments = [], [], []
        self.next_job, self.last_time = 0, -1
        self.completed = set()

    def advance(self, now):
        require(now >= self.last_time, "simulation time moved backwards")
        self.last_time = now
        while self.events and self.events[0][0] <= now:
            _, job_id, server, record = heapq.heappop(self.events)
            require(self.nodes[server].release_job(record), "release of an inactive job")
            self.nodes[server].check_rsrc()
            self.completed.add(job_id)
        while self.next_job < len(self.jobs) and self.jobs[self.next_job]["submit"] <= now:
            self.pending.append(self.jobs[self.next_job])
            self.next_job += 1
        self.pending.sort(key=lambda job: job["job_id"])

    def observation(self):
        free_cpu = np.array([node.idl_cpus for node in self.nodes], dtype=np.int64)
        free_gpu = np.array([node.idl_gpus for node in self.nodes], dtype=np.int64)
        req_cpu = np.array([job["cpu_units"] for job in self.pending], dtype=np.int64)
        req_gpu = np.array([job["gpu_units"] for job in self.pending], dtype=np.int64)
        legal = (req_cpu[:, None] <= free_cpu) & (req_gpu[:, None] <= free_gpu)
        return legal, free_cpu, free_gpu, req_cpu, req_gpu

    def apply(self, job_indices, server_indices, now, legal):
        require(len(job_indices) == len(server_indices), "unequal assignment arrays")
        require(len(set(map(int, job_indices))) == len(job_indices), "duplicate assigned job")
        require(len(set(map(int, server_indices))) == len(server_indices), "server receives >1 new job/tick")
        selected = set()
        for ji, si in zip(job_indices, server_indices):
            ji, si = int(ji), int(si)
            require(legal[ji, si], "illegal CPU/GPU edge")
            job = self.pending[ji]
            require(job["submit"] <= now, "unseen job assigned")
            wait = now - job["submit"]
            record = dict(job_id=job["job_id"], num_cpu=job["cpu_units"], num_gpu=job["gpu_units"])
            require(self.nodes[si].alloc_job(record), "claimed legal allocation exceeds capacity")
            finish = now + job["duration"]
            heapq.heappush(self.events, (finish, job["job_id"], si, record))
            wait_units = WAIT_UNITS_PER_SECOND * wait
            self.assignments.append(dict(job_id=job["job_id"], server_index=si,
                machine_id=self.machines[si]["machine_id"], start=now, finish=finish,
                wait_seconds=wait, **pricing(job), wait_units=wait_units,
                wait_penalty=wait_units / DENOMINATOR,
                reward_units=job["contract_units"] - wait_units,
                reward=(job["contract_units"] - wait_units) / DENOMINATOR))
            selected.add(job["job_id"])
        self.pending = [job for job in self.pending if job["job_id"] not in selected]

    def check(self):
        for node in self.nodes:
            node.check_rsrc()
            require(node.job_cpus == sum(job["num_cpu"] for job in node.job_runn_list)
                    and node.job_gpus == sum(job["num_gpu"] for job in node.job_runn_list),
                    "running-job ledger differs from resource reservation")
        require(len(self.assignments) == len({a["job_id"] for a in self.assignments}), "job assigned twice")


class PatienceEnvironment(BatchEnvironment):
    """Only this private environment sidecar reads job cancellation deadlines."""
    def __init__(self, machines, jobs, patience):
        super().__init__(machines, jobs)
        self._patience = {int(job_id): dict(record) for job_id, record in patience.items()}
        self.cancelled_job_ids, self.cancellations = set(), []

    def advance(self, now):
        super().advance(now)
        expired = []
        for job in self.pending:
            require(job["job_id"] in self._patience, "visible job has no private patience")
            if self._patience[job["job_id"]]["deadline"] <= now:
                expired.append(job)
        expired.sort(key=lambda j: (self._patience[j["job_id"]]["deadline"], j["job_id"]))
        for job in expired:
            deadline = self._patience[job["job_id"]]["deadline"]
            self.cancellations.append(dict(job_id=job["job_id"], submit=job["submit"], cancel_time=deadline,
                wait_seconds=deadline - job["submit"], contract_value=job["contract_value"],
                cpu_units=job["cpu_units"], gpu_units=job["gpu_units"]))
            self.cancelled_job_ids.add(job["job_id"])
        self.pending = [job for job in self.pending if job["job_id"] not in self.cancelled_job_ids]

    def apply(self, job_indices, server_indices, now, legal):
        require(all(self._patience[self.pending[int(ji)]["job_id"]]["deadline"] > now
                    for ji in job_indices), "expired job cannot be admitted")
        super().apply(job_indices, server_indices, now, legal)

    def check(self):
        super().check()
        require(len(self.cancellations) == len(self.cancelled_job_ids), "duplicate cancellation")
        require(self.cancelled_job_ids.isdisjoint(job["job_id"] for job in self.pending), "cancelled job queued")
        require(self.cancelled_job_ids.isdisjoint(a["job_id"] for a in self.assignments), "started job cancelled")


def snapshot(env, start, reference):
    return dict(time=start, reference=reference,
        background_assignments=[a for a in env.assignments if a["start"] < start < a["finish"]],
        initial_queue_ids=[job["job_id"] for job in env.pending if job["submit"] < start],
        pending_job_ids=[job["job_id"] for job in env.pending], next_job=env.next_job,
        free_cpu=[node.idl_cpus for node in env.nodes], free_gpu=[node.idl_gpus for node in env.nodes])


def restore(machines, jobs, initial, patience):
    """Restore common physical background without rerunning it each episode."""
    env = PatienceEnvironment(machines, jobs, patience)
    by_id = {job["job_id"]: job for job in jobs}
    env.pending = [by_id[job_id] for job_id in initial["pending_job_ids"]]
    env.next_job, env.last_time = initial["next_job"], initial["time"]
    env.assignments = copy.deepcopy(initial["background_assignments"])
    for job_id in initial["initial_queue_ids"]:
        require(patience[job_id]["conditional_on_survived_to_score_start"]
                and patience[job_id]["deadline"] > initial["time"], "invalid carry-in queue patience")
    for assignment in env.assignments:
        job_id, server = assignment["job_id"], assignment["server_index"]
        job = by_id[job_id]
        require(assignment["start"] < initial["time"] < assignment["finish"], "inactive background job")
        record = dict(job_id=job_id, num_cpu=job["cpu_units"], num_gpu=job["gpu_units"])
        require(env.nodes[server].alloc_job(record), "background exceeds capacity")
        heapq.heappush(env.events, (assignment["finish"], job_id, server, record))
    env.check()
    require([node.idl_cpus for node in env.nodes] == initial["free_cpu"], "restored CPU differs")
    require([node.idl_gpus for node in env.nodes] == initial["free_gpu"], "restored GPU differs")
    return env
