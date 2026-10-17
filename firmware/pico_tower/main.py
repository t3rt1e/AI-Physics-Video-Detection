# Galileo Tower: reference firmware for a Raspberry Pi Pico (MicroPython).
#
# NOT YET RUN ON HARDWARE. Pin numbers are placeholders: wire, then edit the
# CONFIG block. It speaks the serial contract from the game plan, so the
# laptop side (python -m galileo serve --tower PORT) works unchanged.
#
#   TOWER -> LAPTOP  ARM,id  REL,id,t_us  GATE,id,k,h_mm,t_us  TOWERG,id,g,sigma  BOUNCE,id,n,t_us
#   LAPTOP -> TOWER  VIDEOG,id,g,sigma,verdict   SHOW,text
#
# Copy to the Pico as main.py (Thonny or mpremote: mpremote cp main.py :main.py).

import sys
import time
import select
from machine import Pin, SPI, ADC
import framebuf

try:
    import neopixel
except ImportError:
    neopixel = None

# ---- CONFIG ---------------------------------------------------------------
MAGNET_PIN = 15          # logic-level MOSFET gate (HIGH = magnet on)
BUTTON_PIN = 14          # arcade button to GND, internal pull-up
GATE_PINS = (2, 3, 4, 5) # IR break-beam receivers, LOW while the beam is broken (check yours)
GATE_H_MM = (300.0, 600.0, 900.0, 1200.0)  # measured: ball bottom at magnet -> beam
NEO_PIN = 16             # WS2812B strip for the countdown
NEO_COUNT = 30
PIEZO_ADC = 26           # piezo disc through a divider/clamp into ADC0 (GP26)
PIEZO_THRESHOLD = 9000   # raw 16-bit reading; tune on the bench
SPI_ID, SPI_SCK, SPI_MOSI, MAX_CS = 0, 18, 19, 17
MAX_MODULES = 16         # 8x8 matrices in one chain (four 8x32 boards = 16)
COUNTDOWN_S = 3
GATE_TIMEOUT_MS = 1500
BOUNCE_WINDOW_MS = 2500
MAGNET_MAX_ON_S = 180    # safety: magnet off if nobody drops for 3 minutes
# ---------------------------------------------------------------------------

magnet = Pin(MAGNET_PIN, Pin.OUT, value=0)
button = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_UP)
gates = [Pin(p, Pin.IN, Pin.PULL_UP) for p in GATE_PINS]
piezo = ADC(PIEZO_ADC)
strip = neopixel.NeoPixel(Pin(NEO_PIN), NEO_COUNT) if neopixel else None

gate_times = [0] * len(gates)
gate_hit = [False] * len(gates)


def make_gate_irq(k):
    def handler(pin):
        if not gate_hit[k]:
            gate_times[k] = time.ticks_us()
            gate_hit[k] = True
    return handler


for k, g in enumerate(gates):
    g.irq(trigger=Pin.IRQ_FALLING, handler=make_gate_irq(k))


# ---- MAX7219 chain (text via MicroPython's built-in 8x8 font) -------------
class Matrix:
    def __init__(self, n):
        self.n = n
        self.spi = SPI(SPI_ID, baudrate=8_000_000, polarity=0, phase=0,
                       sck=Pin(SPI_SCK), mosi=Pin(SPI_MOSI))
        self.cs = Pin(MAX_CS, Pin.OUT, value=1)
        self.buf = bytearray(8 * n)
        self.fb = framebuf.FrameBuffer(self.buf, 8 * n, 8, framebuf.MONO_HLSB)
        for reg, val in ((0x0F, 0), (0x09, 0), (0x0B, 7), (0x0A, 4), (0x0C, 1)):
            self._all(reg, val)

    def _all(self, reg, val):
        self.cs(0)
        for _ in range(self.n):
            self.spi.write(bytes((reg, val)))
        self.cs(1)

    def text(self, s):
        self.fb.fill(0)
        self.fb.text(s[: self.n], 0, 0, 1)
        self.show()

    def show(self):
        row_bytes = self.n
        for y in range(8):
            self.cs(0)
            for m in range(self.n - 1, -1, -1):   # last module first in the chain
                self.spi.write(bytes((y + 1, self.buf[y * row_bytes + m])))
            self.cs(1)


try:
    display = Matrix(MAX_MODULES)
except Exception:  # no display wired yet
    display = None


def show_text(s):
    if display:
        display.text(s)


def strip_fill(color, upto=None):
    if not strip:
        return
    n = NEO_COUNT if upto is None else upto
    for i in range(NEO_COUNT):
        strip[i] = color if i < n else (0, 0, 0)
    strip.write()


def send(line):
    sys.stdout.write(line + "\n")


# ---- gravity from the gates: h = c0 + c1 t + (g/2) t^2 --------------------
def fit_g(hs_m, ts_s):
    n = len(hs_m)
    tm = sum(ts_s) / n
    t = [x - tm for x in ts_s]
    s0, s1, s2, s3, s4 = n, sum(t), sum(x * x for x in t), sum(x ** 3 for x in t), sum(x ** 4 for x in t)
    b0 = sum(hs_m)
    b1 = sum(h * x for h, x in zip(hs_m, t))
    b2 = sum(h * x * x for h, x in zip(hs_m, t))
    A = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]]
    B = [b0, b1, b2]
    # Gaussian elimination on the 3x3 normal equations.
    for i in range(3):
        p = A[i][i]
        for j in range(i + 1, 3):
            f = A[j][i] / p
            for c in range(i, 3):
                A[j][c] -= f * A[i][c]
            B[j] -= f * B[i]
    c = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        c[i] = (B[i] - sum(A[i][j] * c[j] for j in range(i + 1, 3))) / A[i][i]
    g = 2 * c[2]
    sigma = 0.0
    if n > 3:
        res = [h - (c[0] + c[1] * x + c[2] * x * x) for h, x in zip(hs_m, t)]
        s2r = sum(r * r for r in res) / (n - 3)
        # Rough: sigma of c2 from the residual scale and the spread of t^2.
        mean_t2 = s2 / n
        var_t2 = s4 / n - mean_t2 ** 2
        sigma = 2 * (s2r / max(n * var_t2, 1e-12)) ** 0.5
    return g, sigma


# ---- laptop messages --------------------------------------------------------
poll = select.poll()
poll.register(sys.stdin, select.POLLIN)
rx = ""
last_tower_g = None


def check_serial():
    global rx
    while poll.poll(0):
        ch = sys.stdin.read(1)
        if ch in ("\n", "\r"):
            handle(rx.strip())
            rx = ""
        else:
            rx += ch


def handle(line):
    if line.startswith("VIDEOG,"):
        parts = line.split(",")
        try:
            g, verdict = float(parts[2]), parts[4]
        except (IndexError, ValueError):
            return
        tg = "%.2f" % last_tower_g if last_tower_g else "----"
        show_text("T%s V%.2f %s" % (tg, g, verdict[:4]))
    elif line.startswith("SHOW,"):
        show_text(line[5:])


# ---- one drop ---------------------------------------------------------------
drop_id = 0


def run_drop():
    global drop_id, last_tower_g
    drop_id += 1
    send("ARM,%d" % drop_id)
    for i in range(COUNTDOWN_S, 0, -1):
        show_text("   %d" % i)
        strip_fill((60, 20, 0), upto=NEO_COUNT * i // COUNTDOWN_S)
        t_end = time.ticks_add(time.ticks_ms(), 1000)
        while time.ticks_diff(t_end, time.ticks_ms()) > 0:
            check_serial()
    for k in range(len(gates)):
        gate_hit[k] = False
    strip_fill((0, 0, 0))
    t_rel = time.ticks_us()
    magnet.value(0)                       # release
    # Times go out on a per-drop clock: REL is always 1,000,000 us and every
    # later event is measured from it, so the Pico's 2^30 us tick wrap can
    # never split a drop. The laptop only uses differences.
    T0 = 1_000_000
    send("REL,%d,%d" % (drop_id, T0))
    deadline = time.ticks_add(time.ticks_ms(), GATE_TIMEOUT_MS)
    while not all(gate_hit) and time.ticks_diff(deadline, time.ticks_ms()) > 0:
        pass
    hs, ts = [], []
    for k in range(len(gates)):
        if gate_hit[k]:
            send("GATE,%d,%d,%.1f,%d" % (drop_id, k + 1, GATE_H_MM[k],
                                         T0 + time.ticks_diff(gate_times[k], t_rel)))
            hs.append(GATE_H_MM[k] / 1000.0)
            ts.append(time.ticks_diff(gate_times[k], t_rel) / 1e6)
    if len(hs) >= 3:
        g, sigma = fit_g(hs, ts)
        last_tower_g = g
        send("TOWERG,%d,%.3f,%.3f" % (drop_id, g, sigma))
        show_text("T%.2f V ...." % g)
    else:
        show_text("GATE MISS")
    # Bounces on the piezo pad.
    n = 0
    quiet_until = 0
    t_stop = time.ticks_add(time.ticks_ms(), BOUNCE_WINDOW_MS)
    while time.ticks_diff(t_stop, time.ticks_ms()) > 0:
        now = time.ticks_us()
        if piezo.read_u16() > PIEZO_THRESHOLD and time.ticks_diff(now, quiet_until) > 0:
            n += 1
            send("BOUNCE,%d,%d,%d" % (drop_id, n, T0 + time.ticks_diff(now, t_rel)))
            quiet_until = time.ticks_add(now, 30_000)   # 30 ms refractory
        check_serial()


def main():
    show_text("GALILEO")
    magnet.value(1)
    on_since = time.ticks_ms()
    while True:
        check_serial()
        if button.value() == 0:
            time.sleep_ms(30)                    # debounce
            if button.value() == 0:
                while button.value() == 0:
                    check_serial()
                if not magnet.value():
                    magnet.value(1)              # re-arm after a safety cut-off
                    show_text("LOAD BALL")
                else:
                    run_drop()
                    time.sleep_ms(1500)
                    magnet.value(1)              # ready for the next ball
                    show_text("LOAD BALL")
                on_since = time.ticks_ms()
        if magnet.value() and time.ticks_diff(time.ticks_ms(), on_since) > MAGNET_MAX_ON_S * 1000:
            magnet.value(0)                      # safety: don't cook the coil
            show_text("PRESS TO ARM")


main()
