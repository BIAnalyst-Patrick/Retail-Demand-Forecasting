"""
Generates a synthetic but behaviorally realistic daily retail sales series
into data/daily_sales.csv, covering Jan 2022 - Jun 2025.

Patterns intentionally built in so forecasting models have real signal to
find:
- Gradual upward revenue trend (business growth).
- Weekly seasonality (weekend lift).
- Annual seasonality (Nov/Dec holiday peak, Jan/Feb post-holiday dip,
  summer secondary bump).
- Discrete promo days with a sales lift, flagged for transparency.
- Random noise on top of all of the above.
"""
import csv
import random
from datetime import date, timedelta

random.seed(7)

START = date(2022, 1, 1)
END = date(2025, 6, 30)

BASE_LEVEL = 4000.0
ANNUAL_GROWTH_RATE = 0.18  # ~18% revenue growth per year


def weekly_seasonality(d):
    # Mon=0 ... Sun=6; weekends run higher
    factors = [0.92, 0.90, 0.93, 0.97, 1.08, 1.25, 1.18]
    return factors[d.weekday()]


def annual_seasonality(d):
    month = d.month
    factors = {
        1: 0.80, 2: 0.82, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.05,
        7: 1.08, 8: 1.05, 9: 0.98, 10: 1.05, 11: 1.35, 12: 1.55,
    }
    return factors[month]


def is_promo_day(d):
    # Black Friday (4th Friday of Nov), Christmas week, New Year's week,
    # plus recurring mid-month promos.
    if d.month == 11 and d.weekday() == 4 and 22 <= d.day <= 28:
        return True
    if d.month == 12 and 20 <= d.day <= 26:
        return True
    if d.month == 1 and d.day <= 3:
        return True
    if d.day in (15, 16):
        return random.random() < 0.5
    return False


def trend_multiplier(d):
    years_elapsed = (d - START).days / 365.25
    return (1 + ANNUAL_GROWTH_RATE) ** years_elapsed


def main():
    rows = []
    d = START
    while d <= END:
        level = BASE_LEVEL * trend_multiplier(d)
        level *= weekly_seasonality(d)
        level *= annual_seasonality(d)

        promo = is_promo_day(d)
        if promo:
            level *= random.uniform(1.25, 1.6)

        noise = random.gauss(0, level * 0.06)
        sales = max(0, round(level + noise, 2))

        rows.append((d.isoformat(), sales, int(promo)))
        d += timedelta(days=1)

    with open("daily_sales.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "sales", "promo_flag"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} daily rows from {START} to {END}")


if __name__ == "__main__":
    main()
