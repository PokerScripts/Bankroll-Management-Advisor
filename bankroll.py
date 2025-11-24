#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bankroll.py - советник по банкролл-менеджменту.
"""

import argparse
import json
import math
import sys
import random
from typing import List, Tuple, Dict

try:
    import numpy as np
except ImportError:
    np = None

# ---------------- Normal CDF ----------------
def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# ---------------- Compute mu, sigma ----------------
def compute_mu_sigma_cash(winrate_bb100, stdev_bb100, hands_per_session, bb_value):
    mu_bb = winrate_bb100 * hands_per_session / 100
    sigma_bb = stdev_bb100 * math.sqrt(hands_per_session / 100)
    return mu_bb * bb_value, sigma_bb * bb_value

def compute_mu_sigma_tournament(roi, buyin, stdev_buyins):
    mu = roi * buyin
    sigma = stdev_buyins * buyin
    return mu, sigma

# ---------------- Risk of Ruin ----------------
def risk_of_ruin_infinite(bankroll, mu, sigma):
    if mu <= 0:
        return 1.0
    return math.exp(-2 * mu * bankroll / (sigma ** 2))

def risk_of_ruin_finite(bankroll, mu, sigma, horizon):
    mean_final = bankroll + mu * horizon
    std_final = sigma * math.sqrt(horizon)
    return normal_cdf((0 - mean_final) / std_final)

# ---------------- Required bankroll ----------------
def required_bankroll_for_risk_infinite(mu, sigma, risk):
    return -(sigma ** 2) / (2 * mu) * math.log(risk)

def find_required_bankroll_finite(mu, sigma, risk, horizon, max_br=1_000_000, tol=1e-6):
    lo, hi = 0, max_br
    for _ in range(100):
        mid = (lo + hi) / 2
        r = risk_of_ruin_finite(mid, mu, sigma, horizon)
        if r > risk:
            lo = mid
        else:
            hi = mid
    return hi

# ---------------- Simulation ----------------
def run_simulations(bankroll0, mu, sigma, horizon, trials, seed=None):
    if seed is not None:
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)

    ruin_count = 0
    final_bankrolls = []
    max_drawdowns = []

    for _ in range(trials):
        br = bankroll0
        peak = br
        ruined = False
        for _ in range(horizon):
            x = random.gauss(mu, sigma)
            br += x
            if br <= 0 and not ruined:
                ruined = True
                br = 0
            peak = max(peak, br)
        if ruined:
            ruin_count += 1
        final_bankrolls.append(br)
        max_drawdowns.append(min(0, br - peak))

    def percentile(data, p):
        k = int(len(data) * p/100)
        return sorted(data)[k]

    return {
        "ror": ruin_count / trials,
        "mean_final": sum(final_bankrolls)/trials,
        "median_final": percentile(final_bankrolls, 50),
        "p10_final": percentile(final_bankrolls, 10),
        "p90_final": percentile(final_bankrolls, 90),
        "median_dd": percentile(max_drawdowns, 50),
        "p90_dd": percentile(max_drawdowns, 90),
        "final_bankrolls": final_bankrolls
    }

# ---------------- Handlers ----------------
def handle_analyze_limit(args):
    if args.game_type == "cash":
        mu, sigma = compute_mu_sigma_cash(args.winrate_bb100, args.stdev_bb100, args.hands_per_session, args.bb_value)
    else:
        mu, sigma = compute_mu_sigma_tournament(args.roi, args.buyin, args.stdev_buyins)

    print("=== Bankroll Analysis ===")
    print(f"Game type: {args.game_type.upper()}")
    print(f"mu: {mu:.2f}, sigma: {sigma:.2f}")
    print(f"Current bankroll: {args.bankroll:.2f}")

    if args.ror_mode == "infinite":
        ror = risk_of_ruin_infinite(args.bankroll, mu, sigma)
        print(f"Estimated RoR: {ror*100:.2f}%")
        if mu > 0:
            req = required_bankroll_for_risk_infinite(mu, sigma, args.risk)
            print(f"Required bankroll for target risk: {req:.2f}")
        else:
            print("Negative EV → RoR = 100%")
    else:
        ror = risk_of_ruin_finite(args.bankroll, mu, sigma, args.horizon)
        print(f"Estimated RoR (finite horizon): {ror*100:.2f}%")
        req = find_required_bankroll_finite(mu, sigma, args.risk, args.horizon)
        print(f"Required bankroll for finite horizon: {req:.2f}")


def handle_suggest_limits(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    gt = cfg["game_type"]
    bankroll = cfg["bankroll"]
    risk = cfg["risk"]
    ror_mode = cfg.get("ror_mode", "infinite")

    print("=== Limits Suggestion ===")

    for lim in cfg["limits"]:
        if gt == "cash":
            mu, sigma = compute_mu_sigma_cash(lim["winrate_bb100"], lim["stdev_bb100"], lim.get("hands_per_session", 1000), lim["bb_value"])
        else:
            mu, sigma = compute_mu_sigma_tournament(lim["roi"], lim["buyin"], lim["stdev_buyins"])

        if ror_mode == "infinite":
            ror = risk_of_ruin_infinite(bankroll, mu, sigma)
            req = required_bankroll_for_risk_infinite(mu, sigma, risk) if mu>0 else float('inf')
        else:
            horizon = cfg.get("horizon", 1000)
            ror = risk_of_ruin_finite(bankroll, mu, sigma, horizon)
            req = find_required_bankroll_finite(mu, sigma, risk, horizon)

        status = "SAFE" if ror <= risk else "RISKY"
        print(f"{lim['name']}: RoR={ror*100:.2f}%, Required BR={req:.2f}, Status={status}")


def handle_simulate(args):
    if args.game_type == "cash":
        mu, sigma = compute_mu_sigma_cash(args.winrate_bb100, args.stdev_bb100, args.hands_per_session, args.bb_value)
    else:
        mu, sigma = compute_mu_sigma_tournament(args.roi, args.buyin, args.stdev_buyins)

    res = run_simulations(args.bankroll, mu, sigma, args.horizon, args.trials, args.seed)
    print("=== Monte Carlo Simulation ===")
    print(f"Estimated RoR: {res['ror']*100:.2f}%")
    print(f"Mean final: {res['mean_final']:.2f}")
    print(f"Median final: {res['median_final']:.2f}")
    print(f"10th pct: {res['p10_final']:.2f}, 90th pct: {res['p90_final']:.2f}")
    print(f"Median DD: {res['median_dd']:.2f}, 90th DD: {res['p90_dd']:.2f}")

    if args.export:
        with open(args.export, "w", newline="", encoding="utf-8") as f:
            f.write("trial,final_bankroll\n")
            for i, br in enumerate(res['final_bankrolls']):
                f.write(f"{i},{br}\n")
        print(f"Results exported to {args.export}")


def handle_config_example(args):
    example = {
        "game_type": "cash",
        "bankroll": 1500,
        "risk": 0.05,
        "ror_mode": "infinite",
        "limits": [
            {
                "name": "NL10",
                "bb_value": 0.1,
                "winrate_bb100": 8,
                "stdev_bb100": 70
            }
        ]
    }
    print(json.dumps(example, indent=2, ensure_ascii=False))

# ---------------- Argparser ----------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Bankroll Advisor")
    sub = p.add_subparsers(dest="command")

    a = sub.add_parser("analyze-limit")
    a.add_argument("--game-type")
    a.add_argument("--bankroll", type=float, required=True)
    a.add_argument("--risk", type=float, default=0.05)
    a.add_argument("--ror-mode", choices=["infinite","finite"], default="infinite")
    a.add_argument("--horizon", type=int, default=1000)
    a.add_argument("--winrate-bb100", type=float)
    a.add_argument("--stdev-bb100", type=float)
    a.add_argument("--hands-per-session", type=float)
    a.add_argument("--bb-value", type=float)
    a.add_argument("--roi", type=float)
    a.add_argument("--buyin", type=float)
    a.add_argument("--stdev-buyins", type=float)

    s = sub.add_parser("suggest-limits")
    s.add_argument("--config", required=True)

    m = sub.add_parser("simulate")
    m.add_argument("--game-type")
    m.add_argument("--bankroll", type=float, required=True)
    m.add_argument("--horizon", type=int, default=500)
    m.add_argument("--trials", type=int, default=1000)
    m.add_argument("--seed", type=int)
    m.add_argument("--export")
    m.add_argument("--winrate-bb100", type=float)
    m.add_argument("--stdev-bb100", type=float)
    m.add_argument("--hands-per-session", type=float)
    m.add_argument("--bb-value", type=float)
    m.add_argument("--roi", type=float)
    m.add_argument("--buyin", type=float)
    m.add_argument("--stdev-buyins", type=float)

    c = sub.add_parser("config-example")

    return p

# ---------------- Main ----------------
def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "analyze-limit":
        handle_analyze_limit(args)
    elif args.command == "suggest-limits":
        handle_suggest_limits(args)
    elif args.command == "simulate":
        handle_simulate(args)
    elif args.command == "config-example":
        handle_config_example(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()