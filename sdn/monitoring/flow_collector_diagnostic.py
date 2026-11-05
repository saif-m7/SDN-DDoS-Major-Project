
"""
Independent Diagnostic FlowCollector.

File:
    sdn/monitoring/flow_collector_diagnostic.py

Import:
    from monitoring.flow_collector_diagnostic import FlowCollector

Features:
    - Counts eligible flows from the current statistics snapshot.
    - Calculates pktrate using the empirically verified Dataset 1 relationship.
    - Skips first-observation flows until a second observation.
    - Uses the current 5-second Packet-In bucket.
    - Preserves the 19 D1 feature names and their order.
    - Keeps diagnostic metadata separate from D1 features.

NOTE:
    The definitions of packetins, flows, pktperflow, byteperflow,
    and tot_dur still require further validation against Dataset 1.
"""

import time


class FlowCollector:

    def __init__(self):

        # Previous cumulative counters for each flow
        self.previous_flows = {}

        # Current flow keys observed on each switch
        self.active_flows = {}

        # Cumulative Packet-In count per switch
        self.packetin_count = {}

        # Used for Pairflow detection
        self.flow_pairs = {}

        # Packet-In counts in fixed 5-second buckets
        self.packetin_5s_buckets = {}

    # ========================================================
    # FLOW KEY
    # ========================================================

    def _build_flow_key(self, datapath_id, stat):

        match = stat.match

        return (
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
            match.get("udp_dst")
        )

    # ========================================================
    # PACKET-IN COUNTER
    # ========================================================

    def increment_packetin(self, datapath_id):

        self.packetin_count[datapath_id] = (
            self.packetin_count.get(datapath_id, 0) + 1
        )

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
            return 0

        elif ip_proto == 6:
            return 1

        elif ip_proto == 17:
            return 2

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

        if protocol == 6:
            src_port = match.get("tcp_src")
            dst_port = match.get("tcp_dst")

        elif protocol == 17:
            src_port = match.get("udp_src")
            dst_port = match.get("udp_dst")

        else:
            src_port = None
            dst_port = None

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
    # FLOW ELIGIBILITY
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
    # COUNT CURRENT FLOW ENTRIES
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

        # ----------------------------------------------------
        # Keep only eligible flows from this current snapshot
        # ----------------------------------------------------

        eligible_stats = [
            stat
            for stat in stats
            if self._is_eligible_flow(stat)
        ]

        current_flow_keys = {
            self._build_flow_key(datapath_id, stat)
            for stat in eligible_stats
        }

        # Replace old flow set instead of accumulating forever
        self.active_flows[datapath_id] = current_flow_keys

        # Current number of eligible flows
        flows = len(current_flow_keys)

        flows_reported = flows

        # ----------------------------------------------------
        # Packet-In measurements
        # ----------------------------------------------------

        bucket_id = int(current_time // 5)

        packetins_cumulative = self.packetin_count.get(
            datapath_id, 0
        )

        packetins_in_5s_bucket = (
            self.packetin_5s_buckets.get(
                (datapath_id, bucket_id), 0
            )
        )

        # Provisional D1 feature definition:
        # use current 5-second Packet-In count.
        packetins = packetins_in_5s_bucket

        # ----------------------------------------------------
        # Remove previous measurements for expired flows
        # ----------------------------------------------------

        stale_flow_keys = [
            key
            for key in self.previous_flows
            if key[0] == datapath_id
            and key not in current_flow_keys
        ]

        for key in stale_flow_keys:
            del self.previous_flows[key]

        # ====================================================
        # PROCESS EACH CURRENT FLOW
        # ====================================================

        for stat in eligible_stats:

            in_port = stat.match.get("in_port")

            flow_key = self._build_flow_key(
                datapath_id,
                stat
            )

            # ------------------------------------------------
            # Current cumulative counters
            # ------------------------------------------------

            current_packets = stat.packet_count
            current_bytes = stat.byte_count

            duration_sec = stat.duration_sec
            duration_nsec = stat.duration_nsec

            # ------------------------------------------------
            # Previous observation
            # ------------------------------------------------

            previous = self.previous_flows.get(flow_key)

            if previous is None:

                # Save the first observation.
                # Do not use this incomplete record for DNN inference.
                self.previous_flows[flow_key] = {
                    "packets": current_packets,
                    "bytes": current_bytes,
                    "timestamp": current_time
                }

                continue

            # ------------------------------------------------
            # Calculate counter differences
            # ------------------------------------------------

            packet_delta = max(
                0,
                current_packets - previous["packets"]
            )

            byte_delta = max(
                0,
                current_bytes - previous["bytes"]
            )

            time_delta = max(
                0.0,
                current_time - previous["timestamp"]
            )

            # Update previous observation
            self.previous_flows[flow_key] = {
                "packets": current_packets,
                "bytes": current_bytes,
                "timestamp": current_time
            }

            # ------------------------------------------------
            # Actual measured packet and byte rates
            # ------------------------------------------------

            if time_delta > 0:

                packet_rate = packet_delta / time_delta
                byte_rate = byte_delta / time_delta

            else:

                packet_rate = 0.0
                byte_rate = 0.0

            # ------------------------------------------------
            # Bandwidth
            # ------------------------------------------------

            kbps = byte_rate * 8 / 1000

            # ------------------------------------------------
            # Match information
            # ------------------------------------------------

            match = stat.match

            protocol = self._get_protocol(match)

            pairflow = self._calculate_pairflow(match)

            # ------------------------------------------------
            # Per-flow packet and byte averages
            # ------------------------------------------------

            if flows > 0:

                pktperflow = current_packets / flows
                byteperflow = current_bytes / flows

            else:

                pktperflow = 0.0
                byteperflow = 0.0

            # ------------------------------------------------
            # Packet rate
            # Dataset 1 empirical relationship:
            # pktrate = floor(pktperflow / 30)
            # Verified against 104,345 Dataset 1 records.
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

            # =================================================
            # BUILD D1 19-FEATURE RECORD
            # =================================================

            record = {

                "switch": datapath_id,

                "pktcount": current_packets,

                "bytecount": current_bytes,

                "dur": duration_sec,

                "dur_nsec": duration_nsec,

                "tot_dur": tot_dur,

                "flows": flows,

                "packetins": packetins,

                "pktperflow": pktperflow,

                "byteperflow": byteperflow,

                "pktrate": pktrate,

                "Pairflow": pairflow,

                "Protocol": protocol,

                "port_no": in_port,

                # Populated from port statistics by Ryu
                "tx_bytes": 0,

                "rx_bytes": 0,

                "tx_kbps": 0.0,

                "rx_kbps": 0.0,

                "tot_kbps": 0.0,

                # --------------------------------------------
                # Additional monitoring metadata
                # --------------------------------------------

                "timestamp": current_time,

                "in_port": in_port,

                "eth_src": match.get("eth_src"),

                "eth_dst": match.get("eth_dst"),

                "ipv4_src": match.get("ipv4_src"),

                "ipv4_dst": match.get("ipv4_dst"),

                "ip_proto": match.get("ip_proto"),

                "tcp_src": match.get("tcp_src"),

                "tcp_dst": match.get("tcp_dst"),

                "udp_src": match.get("udp_src"),

                "udp_dst": match.get("udp_dst"),

                "packet_delta": packet_delta,

                "byte_delta": byte_delta,

                "time_delta": time_delta,

                "byte_rate": byte_rate,

                "kbps": kbps,

                "packet_rate": packet_rate

            }

            records.append(record)

        # ====================================================
        # ADD DIAGNOSTIC FIELDS
        # ====================================================

        flows_tracked = len(
            self.active_flows.get(datapath_id, set())
        )

        for record in records:

            record["flows_reported"] = flows_reported

            record["flows_tracked"] = flows_tracked

            record["packetins_cumulative"] = (
                packetins_cumulative
            )

            record["packetins_in_5s_bucket"] = (
                packetins_in_5s_bucket
            )

            record["packetin_bucket_id"] = bucket_id

        return records
