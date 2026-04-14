
import argparse
import struct
from collections import deque

import numpy as np
import pyqtgraph as pg
import zmq
from PyQt5 import QtCore, QtWidgets

def parse_binary_message(raw, payload_offset, channel_index, channel_count, signal_dtype):
    # Binary FULL packets from falcon start with stream(uint16)+packet(uint64), then AnyType's
    # source timestamp, hardware timestamp, and serial number (24 bytes), then the datatype payload.
    if len(raw) < 34:
        raise ValueError("Packet is too short for binary stream header")

    stream_id, packet_id = struct.unpack_from("<HQ", raw, 0)
    hardware_ts = struct.unpack_from("<Q", raw, 18)[0]

    payload = raw[10 + payload_offset :]
    if not payload:
        return stream_id, packet_id, float(hardware_ts), np.nan

    signal_itemsize = np.dtype(signal_dtype).itemsize
    record_size = 8 + (channel_count * signal_itemsize)
    if len(payload) < record_size or len(payload) % record_size != 0:
        raise ValueError(
            f"Binary payload size {len(payload)} is not compatible with record size {record_size}."
        )

    sample_count = len(payload) // record_size
    signal_offset = (sample_count * 8) + (channel_index * signal_itemsize)
    if signal_offset + signal_itemsize > len(payload):
        raise ValueError("Packet does not contain the requested channel")

    selected = np.frombuffer(payload[signal_offset : signal_offset + signal_itemsize], dtype=signal_dtype)[0]
    return stream_id, packet_id, int(hardware_ts), float(selected)


def parse_phase_message(raw):
    if len(raw) < 8:
        raise ValueError("Phase packet is too short")
    sample_counter, current_phase = struct.unpack_from("<If", raw, 0)
    return int(sample_counter), float(current_phase)


class LivePlotWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(args.address)

        self.phase_socket = self.context.socket(zmq.SUB)
        self.phase_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.phase_socket.connect(args.phase_address)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self.phase_socket, zmq.POLLIN)

        self.setWindowTitle("ZMQ Live Plot (Interleaved Streams)")
        self.resize(1100, 650)

        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.addLegend()
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.setLabel("left", "Value", color="k")
        plot.setLabel("bottom", "Time", units="s", color="k")
        self.setCentralWidget(plot)

        self.curves = {
            0: plot.plot(pen=pg.mkPen((0, 90, 200), width=2), name="orig signal"),
            1: plot.plot(pen=pg.mkPen((200, 80, 0), width=2), name="estimated phase"),
        }
        self.phase_curve = plot.plot(pen=pg.mkPen((20, 140, 20), width=2), name="true phase")

        self.x_data = {
            0: deque(maxlen=args.window_samples),
            1: deque(maxlen=args.window_samples),
        }
        self.y_data = {
            0: deque(maxlen=args.window_samples),
            1: deque(maxlen=args.window_samples),
        }
        self.packet_count = {0: 0, 1: 0}
        self.dropped_packets = {0: 0, 1: 0}
        self.expected_packet_id = {0: None, 1: None}
        self.phase_packet_count = 0

        self.phase_x = deque(maxlen=args.window_samples)
        self.phase_y = deque(maxlen=args.window_samples)
        self.matched_phase_x = deque(maxlen=args.window_samples)
        self.matched_phase_y = deque(maxlen=args.window_samples)
        self.pending_raw = {}
        self.pending_phase = {}
        self.match_count = 0
        self.abs_diff = 0.0
        self.abs_diff_sum = 0.0

        self.status = QtWidgets.QLabel("Waiting for ZMQ packets...")
        self.statusBar().addWidget(self.status)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(args.refresh_ms)

    def append_packet(self, stream_id, x_value, y_value):
        if stream_id not in self.y_data:
            return
        if np.isnan(x_value) or np.isnan(y_value):
            return

        x_scaled = float(x_value) #* self.args.timestamp_scale
        self.x_data[stream_id].append(x_scaled)
        self.y_data[stream_id].append(y_value)

    def _trim_pending(self, store):
        while len(store) > self.args.max_pending_matches:
            oldest_key = next(iter(store))
            del store[oldest_key]

    def add_raw_for_match(self, hardware_ts, value):
        self.pending_raw[hardware_ts] = value
        self.try_match(hardware_ts)

    def add_phase_for_match(self, sample_counter, current_phase):
        self.phase_x.append(float(sample_counter))
        self.phase_y.append(current_phase)
        self.pending_phase[sample_counter] = current_phase
        self._trim_pending(self.pending_phase)

    def try_match(self, key):
        if key not in self.pending_phase:
            return

        raw_value = self.pending_raw.pop(key)
        phase_value = self.pending_phase.pop(key)

        diff = raw_value - phase_value
        self.match_count += 1
        self.abs_diff_sum += abs(diff)
        self.abs_diff = diff
        self.matched_phase_x.append(float(key))
        self.matched_phase_y.append(phase_value)
        # print(f"Matched packet {key}: raw={raw_value:.4f}, phase={phase_value:.4f}, diff={diff:.4f}")

    def update_plot(self):
        received = 0
        while received < self.args.max_packets_per_refresh:
            events = dict(self.poller.poll(timeout=0))
            if self.socket not in events and self.phase_socket not in events:
                break

            if self.socket in events and received < self.args.max_packets_per_refresh:
                raw = self.socket.recv(flags=zmq.NOBLOCK)
                try:
                    stream_id, packet_id, hardware_ts, value = parse_binary_message(
                        raw,
                        self.args.binary_payload_offset,
                        self.args.channel_index,
                        self.args.channel_count,
                        self.args.signal_dtype,
                    )
                except Exception:
                    pass
                else:
                    if stream_id in self.packet_count:
                        self.packet_count[stream_id] += 1

                        if packet_id is not None:
                            expected = self.expected_packet_id[stream_id]
                            if expected is not None and packet_id > expected:
                                missing_packets = packet_id - expected
                                self.dropped_packets[stream_id] += missing_packets

                            # Ignore stale/out-of-order packets for drop accounting.
                            if expected is None or packet_id >= expected:
                                self.expected_packet_id[stream_id] = packet_id + 1

                    self.append_packet(stream_id, hardware_ts, value)
                    if stream_id == 1 and not np.isnan(value):  # add phase value (stream 1)
                        self.add_raw_for_match(hardware_ts, value)
                    received += 1

            if self.phase_socket in events and received < self.args.max_packets_per_refresh:
                phase_raw = self.phase_socket.recv(flags=zmq.NOBLOCK)
                try:
                    sample_counter, current_phase = parse_phase_message(phase_raw)
                except Exception:
                    pass
                else:
                    self.phase_packet_count += 1
                    self.add_phase_for_match(sample_counter, current_phase)
                    received += 1

        for stream_id, curve in self.curves.items():
            curve.setData(np.asarray(self.x_data[stream_id]), np.asarray(self.y_data[stream_id]))
        # self.phase_curve.setData(np.asarray(self.phase_x), np.asarray(self.phase_y))
        self.phase_curve.setData(np.asarray(self.matched_phase_x), np.asarray(self.matched_phase_y))

        mean_abs_diff = self.abs_diff_sum / self.match_count if self.match_count else np.nan

        self.status.setText(
            f"| dropped stream0={self.dropped_packets[0]} stream1={self.dropped_packets[1]} "
            f"| mean error={mean_abs_diff:.4f} last error={self.abs_diff:.4f}"
        )

    def closeEvent(self, event):
        self.timer.stop()
        self.socket.close(linger=0)
        self.phase_socket.close(linger=0)
        self.context.term()
        super().closeEvent(event)


def parse_args():
    parser = argparse.ArgumentParser(description="Live plot two interleaved ZMQ streams.")
    parser.add_argument("--address", default="tcp://localhost:7777", help="ZMQ publisher address")
    parser.add_argument("--phase-address", default="tcp://localhost:5555", help="ZMQ phase publisher address")
    parser.add_argument(
        "--encoding",
        choices=("auto", "binary", "yaml"),
        default="binary",
        help="Packet decoding mode",
    )
    parser.add_argument(
        "--binary-payload-offset",
        type=int,
        default=24,
        help=(
            "Extra bytes to skip after binary stream+packet header. "
            "Use 24 for Falcon FULL/HEADERONLY (source_ts, hardware_ts, serial_number), "
            "use 0 for COMPACT."
        ),
    )
    parser.add_argument(
        "--channel-count",
        type=int,
        default=1,
        help="Number of signal channels stored in each packet payload.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
        help="Channel index to plot from the packet payload.",
    )
    parser.add_argument(
        "--signal-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Numeric type used by the signal payload.",
    )
    parser.add_argument(
        "--timestamp-scale",
        type=float,
        default=1e-6,
        help="Scale factor applied to timestamps before plotting.",
    )
    parser.add_argument(
        "--max-pending-matches",
        type=int,
        default=10000,
        help="Maximum unmatched packets retained for stream-to-phase key matching.",
    )
    parser.add_argument("--window-samples", type=int, default=200, help="Visible points per stream")
    parser.add_argument("--refresh-ms", type=int, default=100, help="GUI update period in milliseconds")
    parser.add_argument(
        "--max-packets-per-refresh",
        type=int,
        default=200,
        help="Packet processing budget each GUI refresh",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.signal_dtype = np.dtype(args.signal_dtype)
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    window = LivePlotWindow(args)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
