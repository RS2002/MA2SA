"""Shared hard-constrained assignment decoder and historical initialization.

No comparison scheduler is exposed. Historical initialization uses the fixed
FIFO/normalized-resource placement rule that generated the experiment's common
starting states; learning episodes use only the learned scores.
"""
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def _fixed_cardinality_assignment(costs, legal, rank):
    """Minimize costs at exactly ``rank`` legal, one-to-one real matches."""
    legal, costs = np.asarray(legal), np.asarray(costs)
    if legal.dtype != np.dtype(bool) or costs.shape != legal.shape:
        raise ValueError("costs and boolean legal must have the same 2D shape")
    transposed = legal.shape[0] > legal.shape[1]
    real_costs, real_legal = (costs.T, legal.T) if transposed else (costs, legal)
    n_rows, n_real_columns = real_costs.shape
    if not 0 <= rank <= n_rows:
        raise ValueError("invalid required matching cardinality")
    matrix = np.empty((n_rows, n_real_columns + n_rows - rank), dtype=np.float64)
    matrix[:, :n_real_columns] = real_costs
    matrix[:, :n_real_columns][~real_legal] = np.inf
    matrix[:, n_real_columns:] = 0.0
    rows, columns = linear_sum_assignment(matrix)
    keep = columns < n_real_columns
    rows, columns = rows[keep], columns[keep]
    if len(rows) != rank:
        raise RuntimeError("assignment violated required cardinality")
    return (columns, rows) if transposed else (rows, columns)


def _history_assignment(legal, free_cpu, free_gpu, req_cpu, req_gpu, jobs):
    """The frozen common-history placement, never a learning/baseline CLI."""
    graph = csr_matrix(legal)
    rank = int(np.count_nonzero(maximum_bipartite_matching(graph, perm_type="column") >= 0))
    if not rank:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    n_jobs, n_servers = legal.shape
    job_match = np.full(n_jobs, -1, dtype=np.int64)
    server_match = np.full(n_servers, -1, dtype=np.int64)
    selected = []
    for root in sorted(range(n_jobs), key=lambda j: (jobs[j]["submit"], jobs[j]["job_id"])):
        if len(selected) == rank:
            break
        parents = np.full(n_servers, -1, dtype=np.int64)
        queue, endpoint = deque([root]), -1
        while queue and endpoint < 0:
            job = queue.popleft()
            for server in graph.indices[graph.indptr[job]:graph.indptr[job + 1]]:
                if parents[server] >= 0:
                    continue
                parents[server] = job
                owner = int(server_match[server])
                if owner < 0:
                    endpoint = int(server)
                    break
                queue.append(owner)
        if endpoint < 0:
            continue
        server = endpoint
        while server >= 0:
            job = int(parents[server])
            previous = int(job_match[job])
            job_match[job], server_match[server] = server, job
            server = previous
        selected.append(root)
    selected = np.asarray(selected, dtype=np.int64)
    costs = (req_cpu[selected, None].astype(np.float64) / 9600
             * (free_cpu[None, :].astype(np.float64) / 9600)
             + req_gpu[selected, None].astype(np.float64) / 800
             * (free_gpu[None, :].astype(np.float64) / 800))
    local, servers = _fixed_cardinality_assignment(costs, legal[selected], rank)
    indices = selected[local]
    order = np.argsort(indices, kind="stable")
    return indices[order], servers[order]
