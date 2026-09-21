"""
core/verify.py — sum_p flow_p == demand

"""

from collections import defaultdict
from core.world import OD


def verify_demand_satisfied(W: dict, flow_per_path: dict, tol: float = 1e-6):

    gp = W["grouped_paths"]

    served = defaultdict(float)
    for (o, d, _pt), entry in flow_per_path.items():
        served[(o, d)] += entry["flow"]

    unreachable = []   
    mismatched  = []  

    for (o, d), demand in OD.items():
        has_path = False
        if o in gp and d in gp[o]:
            for v in gp[o][d]:
                for tr in gp[o][d][v]:
                    if gp[o][d][v][tr]:
                        has_path = True
                        break
                if has_path:
                    break

        s = served.get((o, d), 0.0)
        if not has_path:
            unreachable.append((o, d, demand))
        elif abs(s - demand) > tol:
            mismatched.append((o, d, demand, s))

    total_demand = sum(OD.values())
    total_served = sum(served.values())

    report = {
        "total_demand": total_demand,
        "total_served": total_served,
        "unserved":     total_demand - total_served,
        "unreachable":  unreachable,
        "mismatched":   mismatched,
    }
    ok = (not unreachable) and (not mismatched)
    return ok, report


def print_verification(report: dict) -> None:
    print(f"  [verify] total demand = {report['total_demand']:.2f}")
    print(f"  [verify] total served = {report['total_served']:.4f}")
    print(f"  [verify] unserved     = {report['unserved']:.4f}")

    if report["unreachable"]:
        print(f"  [verify] UNREACHABLE OD pairs ({len(report['unreachable'])}):")
        for o, d, dem in report["unreachable"]:
            print(f"           {o} -> {d}   demand={dem}")

    if report["mismatched"]:
        print(f"  [verify] MISMATCHED OD pairs ({len(report['mismatched'])}):")
        for o, d, dem, s in report["mismatched"]:
            print(f"           {o} -> {d}   demand={dem}  served={s:.4f}")

    if not report["unreachable"] and not report["mismatched"]:
        print("  [verify] OK - all demand satisfied")


def get_unreachable_set(W: dict, flow_per_path: dict) -> set:

    _, report = verify_demand_satisfied(W, flow_per_path)
    return {(o, d) for (o, d, _) in report["unreachable"]}
