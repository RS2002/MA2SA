"""Independent reward, service and requested-capacity metrics."""
from __future__ import annotations
import math
from collections import defaultdict

DENOMINATOR, CALL, CPU, GPU, WAIT = 300000000, 45, 60, 2016, 50000
GAMMA, INTERVAL = .99, 60


def mean(values):
    return math.fsum(values) / len(values) if values else None


def p95(values):
    if not values:
        return None
    values = sorted(values)
    x = .95 * (len(values) - 1)
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (x - lo) * (values[hi] - values[lo])


def close(actual, expected, label):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7), f"{label}: {actual} != {expected}"


def contract_units(job):
    return CALL + (CPU * job["cpu_units"] + GPU * job["gpu_units"]) * job["duration"]


def charge(job):
    return contract_units(job) / DENOMINATOR


def price_check(job):
    expected = contract_units(job)
    assert job["contract_units"] == expected
    for name, value in (("cpu_value", CPU * job["cpu_units"] * job["duration"] / DENOMINATOR),
                        ("gpu_value", GPU * job["gpu_units"] * job["duration"] / DENOMINATOR),
                        ("resource_contract_value", (expected - CALL) / DENOMINATOR),
                        ("invocation_fee", CALL / DENOMINATOR),
                        ("contract_value", expected / DENOMINATOR), ("total_charge", expected / DENOMINATOR)):
        close(job[name], value, f"job {job['job_id']} {name}")


def assignment_check(record, job):
    assert record["finish"] - record["start"] == job["duration"]
    wait = record["start"] - job["submit"]
    assert record["wait_seconds"] == wait >= 0
    assert record["contract_units"] == contract_units(job)
    assert record["wait_units"] == WAIT * wait
    assert record["reward_units"] == contract_units(job) - WAIT * wait
    for name in ("cpu_value", "gpu_value", "resource_contract_value", "invocation_fee", "contract_value", "total_charge"):
        close(record[name], job[name], f"assignment {name}")
    close(record["wait_penalty"], WAIT * wait / DENOMINATOR, "assignment waiting deduction")
    close(record["reward"], record["reward_units"] / DENOMINATOR, "assignment local reward")


def requested_seconds(sequence, by_id, resource, begin, end):
    return math.fsum(by_id[a["job_id"]][f"{resource}_units"] / 100
                     * max(0, min(a["finish"], end) - max(a["start"], begin)) for a in sequence)


def metrics(population, assignments, cancellations, background, by_id, initial_ids,
            begin, end, totals, discount_origin, decisions=()):
    """Population is fixed; outcomes after end are deliberately censored."""
    ids = {j["job_id"] for j in population}
    assert len(ids) == len(population) and all(j["submit"] < end for j in population)
    accepted = {a["job_id"]: a for a in assignments if a["job_id"] in ids and begin <= a["start"] < end}
    cancelled = {a["job_id"]: a for a in cancellations if a["job_id"] in ids and begin <= a["cancel_time"] <= end}
    assert accepted.keys().isdisjoint(cancelled)
    admitted = list(accepted.values())
    cancelled_records = list(cancelled.values())
    completed = [a for a in admitted if a["finish"] <= end]
    running = [a for a in admitted if a["finish"] > end]
    pending = [j for j in population if j["job_id"] not in accepted and j["job_id"] not in cancelled]
    window_wait, age_wait = [], []
    for job in population:
        jid = job["job_id"]
        stop = (accepted[jid]["start"] if jid in accepted else
                cancelled[jid]["cancel_time"] if jid in cancelled else end)
        assert stop >= max(begin, job["submit"])
        window_wait.append(stop - max(begin, job["submit"]))
        age_wait.append(stop - job["submit"])
    inherited = sum(begin - j["submit"] for j in population if j["job_id"] in initial_ids)
    close(math.fsum(age_wait), math.fsum(window_wait) + inherited, "waiting age/window/inherited split")
    count = len(population)
    accepted_units = sum(contract_units(by_id[a["job_id"]]) for a in admitted)
    wait_units = sum(WAIT * a["wait_seconds"] for a in admitted)
    resource_value = (accepted_units - CALL * len(admitted)) / DENOMINATOR
    calls = CALL * len(admitted) / DENOMINATOR
    value = accepted_units / DENOMINATOR
    penalty = wait_units / DENOMINATOR
    reward_units = accepted_units - wait_units
    accepted_wait = [a["wait_seconds"] for a in admitted]
    cancelled_wait = [a["wait_seconds"] for a in cancelled_records]
    completion_times = [a["finish"] - by_id[a["job_id"]]["submit"] for a in completed]
    negative = sum(contract_units(by_id[a["job_id"]]) - WAIT * a["wait_seconds"] < 0 for a in admitted)
    offered = sum(contract_units(j) for j in population) / DENOMINATOR
    cancelled_value = sum(contract_units(by_id[a["job_id"]]) for a in cancelled_records) / DENOMINATOR
    pending_value = sum(contract_units(j) for j in pending) / DENOMINATOR
    close(offered, value + cancelled_value + pending_value, "offered opportunity value partition")
    output = dict(
        observation_start=begin, observation_end=end, horizon_seconds=end - begin,
        arrived_jobs=count, arrived_instances=sum(j["num_inst"] for j in population),
        initial_queued_jobs=sum(j["job_id"] in initial_ids for j in population),
        new_arrived_jobs=sum(j["job_id"] not in initial_ids for j in population),
        accepted_jobs=len(admitted), completed_jobs=len(completed), running_jobs=len(running),
        cancelled_jobs=len(cancelled_records), pending_jobs=len(pending), queued_jobs=len(pending),
        contract_value=value, total_charge=value, resource_contract_value=resource_value,
        invocation_fee=calls, wait_penalty=penalty, reward=reward_units / DENOMINATOR,
        reward_units=reward_units, contract_units=accepted_units, wait_units=wait_units,
        cpu_value=sum(CPU * by_id[a["job_id"]]["cpu_units"] * by_id[a["job_id"]]["duration"] for a in admitted) / DENOMINATOR,
        gpu_value=sum(GPU * by_id[a["job_id"]]["gpu_units"] * by_id[a["job_id"]]["duration"] for a in admitted) / DENOMINATOR,
        discounted_reward=math.fsum((GAMMA ** ((a["start"] - discount_origin) // INTERVAL))
                                   * (contract_units(by_id[a["job_id"]]) - WAIT * a["wait_seconds"]) / DENOMINATOR for a in admitted),
        discount_gamma=GAMMA, discount_origin_seconds=discount_origin, discount_interval_seconds=INTERVAL,
        negative_reward_accepted_jobs=negative,
        negative_reward_accepted_rate=negative / len(admitted) if admitted else None,
        negative_reward_population_rate=negative / count if count else None,
        offered_contract_value=offered, cancelled_contract_value=cancelled_value, pending_contract_value=pending_value,
        queue_wait_seconds_total=math.fsum(window_wait), restricted_wait_seconds_total=math.fsum(window_wait),
        queue_wait_age_seconds_total=math.fsum(age_wait),
        initial_queue_inherited_wait_seconds_total=inherited,
        restricted_wait_seconds_mean=mean(window_wait), restricted_wait_seconds_p95=p95(window_wait),
        observed_wait_age_seconds_mean=mean(age_wait), observed_wait_age_seconds_p95=p95(age_wait),
        queue_jobs_time_average=math.fsum(window_wait) / (end - begin),
        accepted_wait_seconds_mean=mean(accepted_wait), accepted_wait_seconds_p95=p95(accepted_wait),
        cancelled_wait_seconds_mean=mean(cancelled_wait), cancelled_wait_seconds_p95=p95(cancelled_wait),
        cancelled_wait_seconds_total=math.fsum(cancelled_wait),
        pending_wait_age_seconds_mean=mean([end - j["submit"] for j in pending]),
        pending_wait_age_seconds_p95=p95([end - j["submit"] for j in pending]),
        completed_turnaround_seconds_mean=mean(completion_times), completed_turnaround_seconds_p95=p95(completion_times),
        completed_turnaround_population="completed admitted subset; cancellations and unfinished jobs excluded only from this labelled subset metric",
        waiting_population="all fixed cohort jobs; waiting stops at admission/cancellation/end, cancellation waiting is retained",
        all_wait_stop_events_observed=not pending,
        opportunity_value_semantics="cancelled/pending charges are unaccepted opportunities, not revenue and not extra reward penalties",
        utilization_semantics="reserved requested resource-time occupancy, not measured effective hardware utilization",
        decision_count=len(decisions), decision_seconds_mean=mean(list(decisions)), decision_seconds_p95=p95(list(decisions)),
    )
    assert len(admitted) + len(cancelled_records) + len(pending) == count
    assert len(completed) + len(running) == len(admitted)
    for name, rows in (("accepted", admitted), ("completed", completed), ("running", running),
                       ("cancelled", cancelled_records), ("pending", pending)):
        output[f"{name}_rate"] = output[f"{name}_jobs"] / count if count else None
        output[f"{name}_instances"] = sum(by_id[a["job_id"]]["num_inst"] for a in rows)
    output["queued_rate"] = output["pending_rate"]
    for resource in ("cpu", "gpu"):
        cohort_use = requested_seconds(admitted, by_id, resource, begin, end)
        background_use = requested_seconds(background, by_id, resource, begin, end)
        for prefix, amount in (("", cohort_use + background_use), ("cohort_", cohort_use), ("background_", background_use)):
            output[f"{prefix}{resource}_requested_resource_seconds"] = amount
            output[f"{prefix}{resource}_requested_occupancy_rate"] = amount / (totals[resource] * (end - begin)) if totals[resource] else None
    inside = (.072 * output["cohort_cpu_requested_resource_seconds"] + 2.4192 * output["cohort_gpu_requested_resource_seconds"]) / 3600
    outside = math.fsum((CPU * by_id[a["job_id"]]["cpu_units"] + GPU * by_id[a["job_id"]]["gpu_units"])
                        * max(0, a["finish"] - max(a["start"], end)) / DENOMINATOR for a in admitted)
    close(resource_value, inside + outside, "resource fee service-time decomposition")
    close(value, inside + calls + outside, "total charge service/invocation/future decomposition")
    close(value - penalty, output["reward"], "local reward financial decomposition")
    output.update(resource_service_value_inside_observation=inside, resource_service_value_after_observation=outside,
                  total_charge_recognized_inside_observation=inside + calls, total_charge_after_observation=outside,
                  contract_value_after_observation_share=outside / value if value else None,
                  background_initial_jobs=len(background),
                  background_completed_jobs=sum(a["finish"] <= end for a in background),
                  background_running_jobs=sum(a["finish"] > end for a in background),
                  background_revenue_counted=False)
    return output


def resource_check(machines, background, assignments, by_id, begin, end):
    """Independent server-capacity envelope, including exact completion times."""
    changes = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for a in background + assignments:
        server = a["server_index"]
        assert isinstance(server, int) and 0 <= server < len(machines)
        assert a.get("machine_id", machines[server]["machine_id"]) == machines[server]["machine_id"]
        job = by_id[a["job_id"]]
        left, right = max(begin, a["start"]), min(end, a["finish"])
        assert right > left
        for dimension, name in enumerate(("cpu", "gpu")):
            changes[server][left][dimension] += job[f"{name}_units"]
            changes[server][right][dimension] -= job[f"{name}_units"]
    for server, events in changes.items():
        cpu = gpu = 0
        for _, delta in sorted(events.items()):
            cpu += delta[0]
            gpu += delta[1]
            assert 0 <= cpu <= machines[server]["cpu_units"], f"CPU capacity violation on server {server}"
            assert 0 <= gpu <= machines[server]["gpu_units"], f"GPU capacity violation on server {server}"
        assert cpu == gpu == 0
    return dict(status="PASS", checked_servers=len(changes), exact_resource_release_times=True)
