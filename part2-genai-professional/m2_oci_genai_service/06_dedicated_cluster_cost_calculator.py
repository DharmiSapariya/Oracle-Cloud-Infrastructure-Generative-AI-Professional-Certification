"""Course lesson: "Dedicated AI Clusters - Sizing and Pricing" as a calculator.

Rules from the course (as recorded - ALWAYS check current Oracle docs for units and prices):
  * Fine-tuning cluster: billed per unit-hour, minimum commitment 1 hour (partial hours round UP).
  * Hosting cluster:     minimum commitment is the WHOLE month = 744 unit-hours; no partial hosting.
  * Cohere models use Cohere unit types (large/small); Llama uses Large Meta units. No mixing families.
  * A T-Few hosting cluster can host up to 50 models (endpoints) on the same cluster.

"Bob's scenario": fine-tune Cohere Command R (08-2024) weekly and host the custom model all month.
  fine-tuning: 8 small-Cohere units x 5 h x 4 runs = 160 unit-hours
  hosting    : 1 unit x 744 h                       = 744 unit-hours
  total      : 904 unit-hours x ~$6.50              = ~$5,876 (~$6,000)

    python .../06_dedicated_cluster_cost_calculator.py                   # Bob's scenario
    python .../06_dedicated_cluster_cost_calculator.py --ft-units 4 --ft-hours 2.5 --runs 2 --host-units 1 --rate 7
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

HOURS_IN_MONTH = 744  # 31 days * 24h - hosting minimum commitment


@dataclass
class CostBreakdown:
    fine_tuning_unit_hours: float
    hosting_unit_hours: float
    rate_per_unit_hour: float

    @property
    def total_unit_hours(self) -> float:
        return self.fine_tuning_unit_hours + self.hosting_unit_hours

    @property
    def total_cost(self) -> float:
        return self.total_unit_hours * self.rate_per_unit_hour


def monthly_cost(
    ft_units: int, ft_hours_per_run: float, runs_per_month: int, host_units: int, rate: float, hosting: bool = True
) -> CostBreakdown:
    billed_hours_per_run = max(1, math.ceil(ft_hours_per_run))  # min 1 hour, full-hour increments
    ft = ft_units * billed_hours_per_run * runs_per_month
    host = host_units * HOURS_IN_MONTH if hosting else 0
    return CostBreakdown(ft, host, rate)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ft-units", type=int, default=8, help="fine-tuning units (Command R 08-2024 needed 8 small Cohere)")
    ap.add_argument("--ft-hours", type=float, default=5, help="hours per fine-tuning run")
    ap.add_argument("--runs", type=int, default=4, help="fine-tuning runs per month")
    ap.add_argument("--host-units", type=int, default=1)
    ap.add_argument("--rate", type=float, default=6.50, help="USD per unit-hour (course example rate)")
    ap.add_argument("--no-hosting", action="store_true")
    a = ap.parse_args()

    c = monthly_cost(a.ft_units, a.ft_hours, a.runs, a.host_units, a.rate, hosting=not a.no_hosting)
    print(f"fine-tuning : {c.fine_tuning_unit_hours:>7.0f} unit-hours")
    print(f"hosting     : {c.hosting_unit_hours:>7.0f} unit-hours")
    print(f"total       : {c.total_unit_hours:>7.0f} unit-hours x ${c.rate_per_unit_hour:.2f} = ${c.total_cost:,.2f}")
    print(f"units to request via service-limit increase: {a.ft_units + a.host_units}")
