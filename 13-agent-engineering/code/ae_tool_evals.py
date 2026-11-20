"""Tool-use evals: score a trajectory, and measure reliability with pass^k.

A trajectory is the list of tool calls an agent made: (tool, args). We compare it to
an expected trajectory. pass@k asks "did at least one of k tries succeed", pass^k asks
"did ALL k tries succeed", which is what a user running the agent every day feels.

Run:  python ae_tool_evals.py
"""
from math import comb


def score_trajectory(expected, actual):
    """Return dict with tool_accuracy, arg_accuracy, order_ok, extra_calls, success.
    Expected and actual are lists of (tool, args_dict)."""
    exp_tools = [t for t, _ in expected]
    act_tools = [t for t, _ in actual]
    matched = 0
    arg_ok = 0
    used = set()
    for tool, args in expected:
        for j, (at, aa) in enumerate(actual):
            if j not in used and at == tool:
                used.add(j)
                matched += 1
                arg_ok += all(aa.get(k) == v for k, v in args.items())
                break
    order = [t for t in act_tools if t in exp_tools]
    first_seen = []
    for t in order:
        if t not in first_seen:
            first_seen.append(t)
    order_ok = first_seen == [t for t in exp_tools if t in first_seen]
    extra = len(actual) - matched
    n = len(expected)
    return {"tool_accuracy": matched / n, "arg_accuracy": arg_ok / n, "order_ok": order_ok,
            "extra_calls": extra, "success": matched == n and arg_ok == n and order_ok}


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n, c, k):
    """Probability that k tries drawn from n (c successes) ALL succeed."""
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


if __name__ == "__main__":
    exp = [("search_orders", {"customer": "c42"}), ("get_order", {"id": 8841}), ("issue_refund", {"id": 8841, "amount": 40})]
    good = [("search_orders", {"customer": "c42"}), ("get_order", {"id": 8841}), ("issue_refund", {"id": 8841, "amount": 40})]
    wrong_arg = [("search_orders", {"customer": "c42"}), ("get_order", {"id": 8841}), ("issue_refund", {"id": 8841, "amount": 400})]
    wrong_order = [("get_order", {"id": 8841}), ("search_orders", {"customer": "c42"}), ("issue_refund", {"id": 8841, "amount": 40})]
    noisy = good[:2] + [("get_order", {"id": 8841}), ("get_order", {"id": 8841})] + good[2:]
    for name, a in (("good", good), ("wrong amount", wrong_arg), ("wrong order", wrong_order), ("noisy retries", noisy)):
        print(f"{name:<14}", score_trajectory(exp, a))
    print("10 runs, 8 succeeded (80 percent)")
    for k in (1, 3, 5, 8):
        print(f"  k={k}: pass@k={pass_at_k(10, 8, k):.4f}   pass^k={pass_hat_k(10, 8, k):.4f}")
