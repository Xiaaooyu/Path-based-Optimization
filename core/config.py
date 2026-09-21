"""
core/config.py — All hyper-parameters and network definitions.
Edit only this file to change experimental settings.
"""

# ── Network ───────────────────────────────────────────────────────────────────
LINE_DEFS: dict[str, list[str]] = {
    "1": ["s1", "s2", "s3"],
    "2": ["s4", "s2", "s5"],
    "3": ["s3", "s2", "s1"],
    "4": ["s5", "s2", "s4"],
}

# ── Operations ────────────────────────────────────────────────────────────────
TRAVEL_TIME:   int   = 1     # time units per hop
doors_per_run: int   = 2     # doors per train
MAX_WAIT:      int   = 6     # max wait units a passenger considers
MAX_XFER_WAIT: int   = 6  
max_capacity:  int   = 40    # max on-train headcount per segment
penalty_weight: float = 50.0  # λ: penalty per unit of capacity violation

# ── Discount levels ───────────────────────────────────────────────────────────
# Sorted descending: pie[0] = largest discount (most attractive),
#                    pie[-1] = smallest discount (least attractive / no discount)
pie:           list[float] = sorted(
    [0.75, 0.5, 0.25, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.0], reverse=True
)
max_pie_index: int = len(pie) - 1   # all doors start here

# ── Utility coefficients ──────────────────────────────────────────────────────
beta0: float =  2.0    # constant
beta1: float = -0.20   # fare discount  (negative: lower disc → lower utility)
beta2: float = -0.01   # door position  (negative: closer → higher utility)
beta3: float =  -0.05   # wait/transfer penalty per unit

# ── Dwell-time coefficients ───────────────────────────────────────────────────
alpha0: float = 4.92
alpha1: float = 0.21
alpha2: float = 0.25
