
"""
Independent Diagnostic FlowCollector.

File:
    src/monitoring/flow_collector_diagnostic.py

Import in ryu_monitor_updated.py:
    from monitoring.flow_collector_diagnostic import FlowCollector

IMPORTANT:
- This is an independent implementation.
- It does not inherit from flow_collector.py.
- Original 19 D1 feature calculations are preserved,
  except for the explicitly updated pktrate calculation.
- Diagnostic fields are added separately.
- Original flow_collector.py remains untouched.
"""

import time


class FlowCollector:

    def __init__(self):

        # Previous cumulative counters for each flow
        self.previous_flows = {}

        # Active flows observed on each switch
        self.active_flows = {}

        # Packet-In counter for each switch
        self.packetin_count = {}

        # Used for Pairflow detection
        self.flow_pairs = {}

        # Diagnostic Packet-In counts in fixed 5-second buckets
        self.packetin_5s_buckets = {}

    # ========================================================
    # FLOW KEY
    # ========================================================

    def _build_flow_key(self, datapath_id, stat):

        match = stat.match

        flow_key = (
            datapath_id,
            match.get("in_port"),
            match.get("eth_src"),
            match.get("eth_dst"),
            match.get("ipv4_src"),
            match.get("ipv4_dst"),
            match.get("ip_proto"),
            match.get("tcp_src"),
            match.get("tcp_dst"),
            match.get("udp_src"),
            match.get("udp_dst"),
        )

        return flow_key

    # ========================================================
    # PACKET-IN COUNTER
    # ========================================================

    def increment_packetin(self, datapath_id):

        if datapath_id not in self.packetin_count:
            self.packetin_count[datapath_id] = 0

        self.packetin_count[datapath_id] += 1

        # --------------------------------------------
        # Additional 5-second diagnostic counter
        # --------------------------------------------

        bucket_id = int(time.time() // 5)

        key = (datapath_id, bucket_id)

        self.packetin_5s_buckets[key] = (
            self.packetin_5s_buckets.get(key, 0) + 1
        )

        # Keep only current and previous buckets

        oldest_bucket = bucket_id - 1

        stale_keys = [
            bucket_key
            for bucket_key in self.packetin_5s_buckets
            if bucket_key[1] < oldest_bucket
        ]

        for bucket_key in stale_keys:
            del self.packetin_5s_buckets[bucket_key]

    # ========================================================
    # PROTOCOL
    # Dataset 1 mapping:
    # ICMP = 0, TCP = 1, UDP = 2
    # ========================================================

    def _get_protocol(self, match):

        ip_proto = match.get("ip_proto")

        if ip_proto == 1:
            return 0       # ICMP

        elif ip_proto == 6:
            return 1       # TCP

        elif ip_proto == 17:
            return 2       # UDP

        # Unknown protocols must not be classified as ICMP
        return None

    # ========================================================
    # PAIRFLOW
    # ========================================================

    def _calculate_pairflow(self, match):

        src_ip = match.get("ipv4_src")
        dst_ip = match.get("ipv4_dst")

        if src_ip is None or dst_ip is None:
            return 0

        protocol = match.get("ip_proto")

        src_port = (
            match.get("tcp_src")
            if protocol == 6
            else match.get("udp_src")
            if protocol == 17
            else None
        )

        dst_port = (
            match.get("tcp_dst")
            if protocol == 6
            else match.get("udp_dst")
            if protocol == 17
            else None
        )

        forward_key = (
            src_ip,
            dst_ip,
            protocol,
            src_port,
            dst_port
        )

        reverse_key = (
            dst_ip,
            src_ip,
            protocol,
            dst_port,
            src_port
        )

        if reverse_key in self.flow_pairs:
            return 1

        self.flow_pairs[forward_key] = True

        return 0

    # ========================================================
    # DIAGNOSTIC FLOW ELIGIBILITY
    # ========================================================

    @staticmethod
    def _is_eligible_flow(stat):

        # Ignore table-miss flow
        if getattr(stat, "priority", None) == 0:
            return False

        # Ignore entries without input port
        if stat.match.get("in_port") is None:
            return False

        return True

    # ========================================================
    # COUNT REPORTED FLOW ENTRIES
    # ========================================================

    @classmethod
    def _count_eligible_flow_entries(cls, stats):

        return sum(
            1
            for stat in stats
            if cls._is_eligible_flow(stat)
        )

    # ========================================================
    # PROCESS FLOW STATISTICS
    # ========================================================

    def process_flow_stats(self, datapath_id, stats):

        current_time = time.time()

        # Materialize statistics once
        stats = list(stats)

        records = []

        # --------------------------------------------
        # Diagnostic: flows reported in this reply
        # --------------------------------------------

        flows_reported = (
            self._count_eligible_flow_entries(stats)
        )

        # --------------------------------------------
        # Diagnostic: Packet-In measurements
        # --------------------------------------------

        bucket_id = int(current_time // 5)

        packetins_cumulative = self.packetin_count.get(
            datapath_id,
            0
        )

        packetins_in_5s_bucket = (
            self.packetin_5s_buckets.get(
                (datapath_id, bucket_id),
                0
            )
        )

        # --------------------------------------------
        # Initialize switch flow storage
        # --------------------------------------------

        if datapath_id not in self.active_flows:
            self.active_flows[datapath_id] = set()

        # ====================================================
        # ORIGINAL FLOW PROCESSING
        # ====================================================

        for stat in stats:

            # ------------------------------------------------
            # Ignore table-miss flow
            # ------------------------------------------------

            if stat.priority == 0:
                continue

            # ------------------------------------------------
            # Input port
            # ------------------------------------------------

            in_port = stat.match.get(
                "in_port",
                None
            )

            if in_port is None:
                continue

            # ------------------------------------------------
            # Flow key
            # ------------------------------------------------

            flow_key = self._build_flow_key(
                datapath_id,
                stat
            )

            self.active_flows[datapath_id].add(
                flow_key
            )

            # ------------------------------------------------
            # Current cumulative counters
            # ------------------------------------------------

            current_packets = stat.packet_count

            current_bytes = stat.byte_count

            duration_sec = stat.duration_sec

            duration_nsec = stat.duration_nsec

            current_duration = (
                duration_sec
                + duration_nsec / 1_000_000_000
            )

            # ------------------------------------------------
            # Previous observation
            # ------------------------------------------------

            if flow_key not in self.previous_flows:

                packet_delta = 0

                byte_delta = 0

                time_delta = 0.0

            else:

                previous = self.previous_flows[
                    flow_key
                ]

                packet_delta = max(
                    0,
                    current_packets -
                    previous["packets"]
                )

                byte_delta = max(
                    0,
                    current_bytes -
                    previous["bytes"]
                )

                time_delta = max(
                    0.0,
                    current_time -
                    previous["timestamp"]
                )

            # ------------------------------------------------
            # Packet rate
            # Retained for diagnostic calculations
            # ------------------------------------------------

            if time_delta > 0:

                packet_rate = (
                    packet_delta /
                    time_delta
                )

                byte_rate = (
                    byte_delta /
                    time_delta
                )

            else:

                packet_rate = 0.0

                byte_rate = 0.0

            # ------------------------------------------------
            # Kbps
            # ------------------------------------------------

            kbps = (
                byte_rate * 8 / 1000
            )

            # ------------------------------------------------
            # Save current counters
            # ------------------------------------------------

            self.previous_flows[flow_key] = {

                "packets": current_packets,

                "bytes": current_bytes,

                "timestamp": current_time
            }

            # ------------------------------------------------
            # Match information
            # ------------------------------------------------

            match = stat.match

            protocol = self._get_protocol(
                match
            )

            pairflow = self._calculate_pairflow(
                match
            )

            # ------------------------------------------------
            # Number of active flows
            # ------------------------------------------------

            flows = len(
                self.active_flows[datapath_id]
            )

            # ------------------------------------------------
            # PacketIn count
            # ------------------------------------------------

            packetins = self.packetin_count.get(
                datapath_id,
                0
            )

            # ------------------------------------------------
            # pktperflow
            #
            # Existing operational live definition:
            # total packets / active flows
            # ------------------------------------------------

            if flows > 0:

                pktperflow = (
                    current_packets /
                    flows
                )

                byteperflow = (
                    current_bytes /
                    flows
                )

            else:

                pktperflow = 0.0

                byteperflow = 0.0

            # ------------------------------------------------
            # UPDATED PKTRATE
            #
            # Empirically observed Dataset 1 relationship:
            # pktrate = pktperflow // 30
            # ------------------------------------------------

            pktrate = int(pktperflow // 30)

            # ------------------------------------------------
            # Total duration
            # ------------------------------------------------

            tot_dur = (
                duration_sec * 1_000_000_000
                + duration_nsec
            )

            # ------------------------------------------------
            # Total bandwidth
            # ------------------------------------------------

            tot_kbps = kbps

            # ------------------------------------------------
            # Build original 19-feature record
            # ------------------------------------------------

            record = {

                # 1
                "switch": datapath_id,

                # 2
                "pktcount": current_packets,

                # 3
                "bytecount": current_bytes,

                # 4
                "dur": duration_sec,

                # 5
                "dur_nsec": duration_nsec,

                # 6
                "tot_dur": tot_dur,

                # 7
                "flows": flows,

                # 8
                "packetins": packetins,

                # 9
                "pktperflow": pktperflow,

                # 10
                "byteperflow": byteperflow,

                # 11
                "pktrate": pktrate,

                # 12
                "Pairflow": pairflow,

                # 13
                "Protocol": protocol,

                # 14
                "port_no": in_port,

                # 15-19
                # Populated from port statistics by Ryu

                "tx_bytes": 0,

                "rx_bytes": 0,

                "tx_kbps": 0.0,

                "rx_kbps": 0.0,

                "tot_kbps": 0.0,

                # ------------------------------------
                # Original monitoring metadata
                # ------------------------------------

                "timestamp": current_time,

                "in_port": in_port,

                "eth_src": match.get(
                    "eth_src"
                ),

                "eth_dst": match.get(
                    "eth_dst"
                ),

                "ipv4_src": match.get(
                    "ipv4_src"
                ),

                "ipv4_dst": match.get(
                    "ipv4_dst"
                ),

                "ip_proto": match.get(
                    "ip_proto"
                ),

                "tcp_src": match.get(
                    "tcp_src"
                ),

                "tcp_dst": match.get(
                    "tcp_dst"
                ),

                "udp_src": match.get(
                    "udp_src"
                ),

                "udp_dst": match.get(
                    "udp_dst"
                ),

                "packet_delta": packet_delta,

                "byte_delta": byte_delta,

                "time_delta": time_delta,

                "byte_rate": byte_rate,

                "kbps": kbps

            }

            records.append(record)

        # ====================================================
        # ADD DIAGNOSTIC FIELDS
        # ====================================================

        # Cumulative number of unique flow keys tracked
        # by this collector for the switch.

        flows_tracked = len(
            self.active_flows.get(
                datapath_id,
                set()
            )
        )

        for record in records:

            # Number of eligible entries in this reply
            record["flows_reported"] = flows_reported

            # Cumulative unique flow keys tracked
            record["flows_tracked"] = flows_tracked

            # Original cumulative Packet-In counter
            record["packetins_cumulative"] = (
                packetins_cumulative
            )

            # Packet-In count in current 5-second bucket
            record["packetins_in_5s_bucket"] = (
                packetins_in_5s_bucket
            )

            # Diagnostic bucket identifier
            record["packetin_bucket_id"] = bucket_id

        return records
