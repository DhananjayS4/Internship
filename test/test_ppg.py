"""
PPG Sensor Diagnostic Test
Run from project root on Raspberry Pi:
    python test/test_ppg.py

Checks:
- Sensor initializes without errors
- FIFO produces new samples
- RED and IR values are non-zero and in physiological range
- Signal has enough variance (light is on and finger is placed)
- BPM estimate is physiologically plausible
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from collections import deque
from drivers.max30102_driver import MAX30102

# ── Config ──────────────────────────────────────────────────────────────────
COLLECT_SEC   = 15      # seconds to collect data
MIN_SIGNAL_RANGE = 500  # minimum ADC range to consider sensor active
SAMPLE_DELAY  = 0.01    # 10ms between polls (sensor runs at ~100sps)

# ── Init ────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  MAX30102 PPG Sensor Diagnostic")
print("=" * 55)

print("\n[1] Initializing sensor...", end=" ", flush=True)
try:
    sensor = MAX30102()
    print("OK ✓")
except Exception as e:
    print(f"FAILED ✗\n    Error: {e}")
    print("\n  Possible causes:")
    print("  - Sensor not wired to I2C bus 1 (GPIO 2/3)")
    print("  - Wrong I2C address (try 0x57 or 0x57)")
    print("  - Run: sudo i2cdetect -y 1  to check")
    sys.exit(1)

# ── Collect ─────────────────────────────────────────────────────────────────
print(f"\n[2] Collecting {COLLECT_SEC}s of PPG data...")
print("    >> Place your FINGER firmly on the sensor now <<\n")

red_buf  = deque(maxlen=1000)
ir_buf   = deque(maxlen=1000)
time_buf = deque(maxlen=1000)
zero_reads = 0
total_reads = 0

start = time.time()
while time.time() - start < COLLECT_SEC:
    red, ir = sensor.read_fifo()
    total_reads += 1
    if ir == 0 and red == 0:
        zero_reads += 1
    else:
        red_buf.append(red)
        ir_buf.append(ir)
        time_buf.append(time.time())

    elapsed = time.time() - start
    # Live display
    bar = int((elapsed / COLLECT_SEC) * 30)
    print(f"\r    [{'█'*bar}{'░'*(30-bar)}] "
          f"IR={ir:>7}  RED={red:>7}  samples={len(ir_buf):>4}", end="", flush=True)
    time.sleep(SAMPLE_DELAY)

print("\n")

# ── Analysis ────────────────────────────────────────────────────────────────
print("[3] Signal Analysis")
print("-" * 40)

ir_arr  = np.array(ir_buf)
red_arr = np.array(red_buf)

if len(ir_arr) == 0:
    print("  ✗ No samples collected at all!")
    print("    Check I2C wiring and sensor power (needs 5V on most breakout boards).")
    sys.exit(1)

ir_mean  = ir_arr.mean()
ir_range = ir_arr.max() - ir_arr.min()
ir_std   = ir_arr.std()

zero_pct = zero_reads / total_reads * 100 if total_reads else 0

print(f"  Total polls     : {total_reads}")
print(f"  Valid samples   : {len(ir_arr)}")
print(f"  Empty reads     : {zero_reads} ({zero_pct:.1f}%)")
print(f"  IR  mean        : {ir_mean:.0f}")
print(f"  IR  range       : {ir_range:.0f}  (min {ir_arr.min()}, max {ir_arr.max()})")
print(f"  IR  std dev     : {ir_std:.0f}")
print(f"  RED mean        : {red_arr.mean():.0f}")

print()

# ── Checks ──────────────────────────────────────────────────────────────────
print("[4] Checks")
print("-" * 40)

ok = True

# Check 1: IR values non-zero
if ir_mean < 1000:
    print("  ✗ IR mean is very low. LED may be OFF.")
    print("    Try: power sensor from 5V pin, not 3.3V.")
    ok = False
else:
    print(f"  ✓ IR mean OK ({ir_mean:.0f})")

# Check 2: Signal has enough range (finger on sensor)
if ir_range < MIN_SIGNAL_RANGE:
    print(f"  ✗ Signal range too low ({ir_range:.0f} < {MIN_SIGNAL_RANGE}).")
    print("    Sensor light may be off, or finger not placed.")
    ok = False
else:
    print(f"  ✓ Signal range OK ({ir_range:.0f})")

# Check 3: Too many empty FIFO reads
if zero_pct > 80:
    print(f"  ✗ {zero_pct:.0f}% of reads returned (0,0). FIFO not filling.")
    print("    Check sensor mode config and LED amplitude registers.")
    ok = False
else:
    print(f"  ✓ FIFO filling rate OK ({100-zero_pct:.0f}% non-empty)")

# Check 4: BPM estimate
if len(time_buf) > 20 and ir_range >= MIN_SIGNAL_RANGE:
    signal = (ir_arr - ir_arr.min()) / (ir_range + 1e-6)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > 0.5 and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)

    filtered = []
    times = list(time_buf)
    for p in peaks:
        if not filtered or (times[p] - times[filtered[-1]]) > 0.4:
            filtered.append(p)

    if len(filtered) >= 2:
        rr = [times[filtered[i]] - times[filtered[i-1]] for i in range(1, len(filtered))]
        bpm = int(60 / np.mean(rr))
        if 30 <= bpm <= 220:
            print(f"  ✓ Estimated BPM: {bpm} (peaks found: {len(filtered)})")
        else:
            print(f"  ✗ BPM out of range: {bpm}. Check signal quality.")
            ok = False
    else:
        print(f"  ⚠ Not enough peaks to estimate BPM ({len(filtered)} found).")
        print("    Press finger more firmly or wait longer.")

print()
print("=" * 40)
if ok:
    print("  ✅ Sensor is WORKING correctly!")
else:
    print("  ❌ Issues detected. See messages above.")
print("=" * 40)
