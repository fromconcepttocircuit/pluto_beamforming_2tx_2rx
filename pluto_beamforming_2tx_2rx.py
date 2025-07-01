#!/usr/bin/env python3

import sys
import numpy as np
import adi
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

try:
    DASH = QtCore.Qt.DashLine
except AttributeError:
    DASH = QtCore.Qt.PenStyle.DashLine


def main() -> None:
    # ─────────── user parameters ───────────
    fc_hz      = 2.5e9
    samp_rate  = 4_000_000
    f_tone     = 637_000

    tx_spacing_m = 0.10
    rx_spacing_m = 0.06

    N          = 2 ** 18
    RX_BUF     = 2 ** 18
    M_avg      = 1
    phase_step = 6
    update_ms  = 200

    f_hp, df_search = 20_000, 50_000
    snr_win   = 10_000
    # ──────────────────────────────────────

    lam   = 3e8 / fc_hz
    d_rx  = rx_spacing_m

    # ────────── SDR init ──────────
    sdr_tx = adi.ad9361("ip:192.168.2.2")
    sdr_rx = adi.ad9361("ip:192.168.2.1")
    for dev in (sdr_tx, sdr_rx):
        dev.sample_rate = samp_rate

    # TX config
    sdr_tx.tx_enabled_channels = [0,1]
    sdr_tx.tx_rf_bandwidth     = int(0.8*samp_rate)
    sdr_tx.tx_lo               = int(fc_hz)
    sdr_tx.tx_hardwaregain_chan0 = -36
    sdr_tx.tx_hardwaregain_chan1 = -36
    sdr_tx.tx_cyclic_buffer    = True

    # RX config
    sdr_rx.rx_enabled_channels = [0,1]
    sdr_rx.rx_rf_bandwidth     = int(0.8*samp_rate)
    sdr_rx.rx_lo               = int(fc_hz)
    sdr_rx.rx_buffer_size      = RX_BUF
    sdr_rx.gain_control_mode   = "manual"
    sdr_rx.rx_gain             = -30
    for a in ("rf_dc_offset_tracking_en","bb_dc_offset_tracking_en"):
        if a in sdr_rx._ctrl.attrs:
            sdr_rx._ctrl.attrs[a].value = "1"
            break

    # ─────────── TX waveform and helper ───────────
    t = np.arange(N)/samp_rate
    base_wave = (2**14)*0.5*np.exp(1j*2*np.pi*f_tone*t)
    def send_tx(phase_deg):
        sdr_tx.tx_destroy_buffer()
        sdr_tx.tx([ base_wave,
                    base_wave * np.exp(1j*np.deg2rad(phase_deg)) ])

    # init at 0°
    send_tx(0.0)

    # ─────────── GUI ───────────
    app = QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    win.resize(950,750)
    vbox = QtWidgets.QVBoxLayout(win)

    # Buttons + phase-edit
    hbox = QtWidgets.QHBoxLayout()
    btn_one  = QtWidgets.QPushButton("One Shot")
    btn_cont = QtWidgets.QPushButton("Continuous")
    btn_cal  = QtWidgets.QPushButton("Calibrate (peak → AoA 0°)")
    btn_lock = QtWidgets.QPushButton("Lock Phase")
    phase_label = QtWidgets.QLabel("Locked Phase (°):")
    phase_edit  = QtWidgets.QLineEdit()
    phase_edit.setFixedWidth(60)
    phase_edit.setEnabled(False)
    for w in (btn_one, btn_cont, btn_cal, btn_lock, phase_label, phase_edit):
        hbox.addWidget(w)
    vbox.addLayout(hbox)

    def make_plot(title, ylab):
        w = pg.GraphicsLayoutWidget()
        p = w.addPlot(title=title)
        p.setLabel("left", ylab)
        p.setLabel("bottom", "TX Phase (°)")
        vbox.addWidget(w)
        return p

    fft_p = make_plot("Received-0 FFT", "Magnitude (dBFS)")
    amp_p = make_plot("Coherent Σ-power", "Σ Power (dB)")
    aoa_p = make_plot("Angle of Arrival", "AoA (°)")

    fft_curve = fft_p.plot(pen="y")
    pk_mark   = pg.ScatterPlotItem(size=12, brush='r'); fft_p.addItem(pk_mark)
    pk_text   = pg.TextItem(color='r', anchor=(0,1)); fft_p.addItem(pk_text)
    snr_text  = pg.TextItem(color='g', anchor=(0,1)); fft_p.addItem(snr_text)

    amp_curve = amp_p.plot(pen="b", symbol="o")
    aoa_curve = aoa_p.plot(pen="c", symbol="t")
    aoa_text  = pg.TextItem(color="w", anchor=(0,1.2)); amp_p.addItem(aoa_text)
    off_lbl   = pg.TextItem("", color="y", anchor=(0,0)); aoa_p.addItem(off_lbl)
    off_lbl.setPos(-175,85)

    amp_p.setLimits(yMin=-300, yMax=10)
    aoa_p.setLimits(yMin=-720, yMax=720)

    win.show()

    # ───────── state ─────────
    mode = "one_shot"   # "one_shot", "continuous", "lock_sweep", "locked"
    cur_phase = -180.0
    aoa_offset = 0.0
    dphi_unwrapped = 0.0

    ph_hist = []
    amp_hist = []
    aoa_raw_hist = []
    aoa_hist = []

    time_hist = []
    amp_time_hist = []
    aoa_time_hist = []
    sample_idx = 0
    locked_phase = 0.0

    # ───────── sweep update ─────────
    def update_sweep():
        nonlocal cur_phase, dphi_unwrapped, mode, locked_phase

        send_tx(cur_phase)

        # capture & FFT
        win_fn = np.hanning(RX_BUF)
        acc0 = np.zeros(RX_BUF, np.complex64)
        acc1 = np.zeros(RX_BUF, np.complex64)
        for _ in range(M_avg):
            x0,x1 = sdr_rx.rx()
            x0 = (x0-x0.mean())/2**11; x1 = (x1-x1.mean())/2**11
            x0 *= win_fn; x1 *= win_fn
            acc0 += np.fft.fftshift(np.fft.fft(x0))/np.sum(win_fn)
            acc1 += np.fft.fftshift(np.fft.fft(x1))/np.sum(win_fn)
        fft0,fft1 = acc0/M_avg, acc1/M_avg
        freq = np.fft.fftshift(np.fft.fftfreq(RX_BUF,1/samp_rate))
        mag0_lin = np.abs(fft0); mag0_db = 20*np.log10(np.maximum(mag0_lin,1e-20))
        mag0_lin[np.abs(freq)<f_hp] = 0
        mask = np.abs(freq - f_tone)<=df_search
        if not mask.any(): return
        idx = int(np.argmax(mag0_lin*mask))
        if idx==0: return

        dphi = np.angle(fft1[idx]/fft0[idx])
        dphi_unwrapped = np.unwrap([dphi_unwrapped, dphi])[-1]

        fft_sum = fft0 + fft1*np.exp(-1j*dphi_unwrapped)
        pow_sum = np.abs(fft_sum)**2

        # FFT plot
        fft_curve.setData(freq, mag0_db)
        pk_f, pk_val = float(freq[idx]), mag0_db[idx]
        pk_mark.setData([pk_f],[pk_val])
        pk_text.setText(f"{pk_f:,.0f} Hz\n{pk_val:5.1f} dBFS")
        pk_text.setPos(pk_f,pk_val)
        S = np.sum(pow_sum[np.abs(freq-pk_f)<=snr_win])
        N = np.sum(pow_sum[~(np.abs(freq-pk_f)<=snr_win)])
        snr_text.setText(f"SNR(Σ) ≈ {10*np.log10(S/max(N,1e-20)):5.1f} dB")
        snr_text.setPos(freq[0], mag0_db.max())

        # power & AoA
        pwr_db = 10*np.log10(np.abs(fft_sum[idx])**2 + 1e-20)
        arg = np.clip((dphi_unwrapped*lam)/(2*np.pi*d_rx), -1,1)
        theta = np.degrees(np.arcsin(arg))
        aoa_val = theta - aoa_offset

        ph_hist.append(cur_phase)
        amp_hist.append(pwr_db)
        aoa_raw_hist.append(theta)
        aoa_hist.append(aoa_val)

        amp_curve.setData(ph_hist, amp_hist)
        aoa_curve.setData(ph_hist, aoa_hist)
        aoa_text.setText(f"AoA ≈ {aoa_val:+6.1f}°")
        aoa_text.setPos(-180, max(amp_hist)+2)

        cur_phase += phase_step
        if cur_phase > 180:
            if mode=="continuous":
                QtCore.QTimer.singleShot(update_ms, start_sweep)
            elif mode=="lock_sweep":
                idx_max = int(np.argmax(amp_hist))
                locked_phase = ph_hist[idx_max]
                phase_edit.setText(f"{locked_phase:.1f}")
                phase_edit.setEnabled(True)
                # reset locked-mode data
                nonlocal sample_idx
                sample_idx = 0
                time_hist.clear(); amp_time_hist.clear(); aoa_time_hist.clear()
                mode = "locked"
                send_tx(locked_phase)
                # switch axes to sample #
                amp_p.setLabel("bottom","Sample #")
                aoa_p.setLabel("bottom","Sample #")
                amp_p.setXRange(0,0)
                aoa_p.setXRange(0,0)
                QtCore.QTimer.singleShot(update_ms, update_locked)
            return

        QtCore.QTimer.singleShot(update_ms, update_sweep)

    # ───────── start/reset sweep ─────────
    def start_sweep():
        nonlocal cur_phase, dphi_unwrapped
        cur_phase = -180.0
        dphi_unwrapped = 0.0
        for lst in (ph_hist, amp_hist, aoa_raw_hist, aoa_hist):
            lst.clear()
        amp_curve.clear(); aoa_curve.clear()
        amp_p.setLabel("bottom","TX Phase (°)")
        aoa_p.setLabel("bottom","TX Phase (°)")
        update_sweep()

    # ───────── proper mode callbacks ─────────
    def on_one_shot():
        nonlocal mode
        mode = "one_shot"
        start_sweep()
    btn_one.clicked.connect(on_one_shot)

    def on_continuous():
        nonlocal mode
        mode = "continuous"
        start_sweep()
    btn_cont.clicked.connect(on_continuous)

    # ───────── calibrate AoA ─────────
    def calibrate():
        nonlocal aoa_offset, aoa_hist
        if not amp_hist: return
        aoa_offset = aoa_raw_hist[int(np.argmax(amp_hist))]
        aoa_hist = [v - aoa_offset for v in aoa_raw_hist]
        aoa_curve.setData(ph_hist, aoa_hist)
        off_lbl.setText(f"Offset: {aoa_offset:+.1f}°")
    btn_cal.clicked.connect(calibrate)

    # ───────── lock-phase sweep start ─────────
    def lock_sweep():
        nonlocal mode
        mode = "lock_sweep"
        start_sweep()
    btn_lock.clicked.connect(lock_sweep)

    # ───────── locked-mode update ─────────
    def update_locked():
        nonlocal sample_idx
        send_tx(locked_phase)

        # capture & FFT
        win_fn = np.hanning(RX_BUF)
        acc0 = np.zeros(RX_BUF, np.complex64)
        acc1 = np.zeros(RX_BUF, np.complex64)
        for _ in range(M_avg):
            x0,x1 = sdr_rx.rx()
            x0 = (x0-x0.mean())/2**11; x1 = (x1-x1.mean())/2**11
            x0 *= win_fn; x1 *= win_fn
            acc0 += np.fft.fftshift(np.fft.fft(x0))/np.sum(win_fn)
            acc1 += np.fft.fftshift(np.fft.fft(x1))/np.sum(win_fn)
        fft0,fft1 = acc0/M_avg, acc1/M_avg
        freq = np.fft.fftshift(np.fft.fftfreq(RX_BUF,1/samp_rate))
        mag0_lin = np.abs(fft0); mag0_db = 20*np.log10(np.maximum(mag0_lin,1e-20))
        mag0_lin[np.abs(freq)<f_hp] = 0
        mask = np.abs(freq - f_tone)<=df_search
        if not mask.any(): return
        idx = int(np.argmax(mag0_lin*mask))
        if idx==0: return

        dphi = np.angle(fft1[idx]/fft0[idx])
        fft_sum = fft0 + fft1*np.exp(-1j*dphi)
        pow_sum = np.abs(fft_sum)**2

        # FFT
        fft_curve.setData(freq, mag0_db)
        pk_f, pk_val = float(freq[idx]), mag0_db[idx]
        pk_mark.setData([pk_f],[pk_val])
        pk_text.setText(f"{pk_f:,.0f} Hz\n{pk_val:5.1f} dBFS")
        pk_text.setPos(pk_f,pk_val)
        S = np.sum(pow_sum[np.abs(freq-pk_f)<=snr_win])
        N = np.sum(pow_sum[~(np.abs(freq-pk_f)<=snr_win)])
        snr_text.setText(f"SNR(Σ) ≈ {10*np.log10(S/max(N,1e-20)):5.1f} dB")
        snr_text.setPos(freq[0], mag0_db.max())

        # power & AoA
        pwr_db = 10*np.log10(np.abs(fft_sum[idx])**2 + 1e-20)
        arg = np.clip((dphi*lam)/(2*np.pi*d_rx), -1,1)
        theta = np.degrees(np.arcsin(arg)) - aoa_offset

        time_hist.append(sample_idx)
        amp_time_hist.append(pwr_db)
        aoa_time_hist.append(theta)

        amp_curve.setData(time_hist, amp_time_hist)
        aoa_curve.setData(time_hist, aoa_time_hist)
        aoa_text.setText(f"AoA ≈ {theta:+6.1f}°")
        aoa_text.setPos(time_hist[0] if time_hist else 0, max(amp_time_hist)+2)

        # auto-scale Y
        if len(amp_time_hist)>3:
            finite = np.array(amp_time_hist)[np.isfinite(amp_time_hist)]
            amp_p.setYRange(finite.min()-3, finite.max()+3)
        if len(aoa_time_hist)>3:
            finite = np.array(aoa_time_hist)[np.isfinite(aoa_time_hist)]
            aoa_p.setYRange(finite.min()-15, finite.max()+15)
        # auto-scale X
        if time_hist:
            amp_p.setXRange(time_hist[0], time_hist[-1])
            aoa_p.setXRange(time_hist[0], time_hist[-1])

        sample_idx += 1
        QtCore.QTimer.singleShot(update_ms, update_locked)

    # ───────── apply edited locked phase ─────────
    def apply_phase_edit():
        nonlocal locked_phase, sample_idx
        try:
            new_ph = float(phase_edit.text())
        except ValueError:
            return
        locked_phase = new_ph
        send_tx(locked_phase)
        sample_idx = 0
        time_hist.clear(); amp_time_hist.clear(); aoa_time_hist.clear()
    phase_edit.returnPressed.connect(apply_phase_edit)

    # initialize and start
    start_sweep()

    if hasattr(app, "exec"):
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
