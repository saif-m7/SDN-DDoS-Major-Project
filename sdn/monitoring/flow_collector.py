import time


class FlowCollector:

    def __init__(self):

        # Previous cumulative counters for each flow
        self.previous_flows = {}

        # Active flows observed on each switch
        self.active_flows = {}

        # PacketIn counter for each switch
        self.packetin_count = {}

        # Used for Pairflow detection
        self.flow_pairs = {}

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

    # ========================================================
    # PROTOCOL
    # ========================================================

    def _get_protocol(self, match):

        ip_proto = match.get("ip_proto")

        if ip_proto == 1:
            return 0       # ICMP

        elif ip_proto == 6:
            return 1       # TCP

        elif ip_proto == 17:
            return 2       # UDP

        return 0

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
    # PROCESS FLOW STATISTICS
    # ========================================================

    def process_flow_stats(self, datapath_id, stats):

        current_time = time.time()

        records = []

        # Initialize switch flow storage
        if datapath_id not in self.active_flows:
            self.active_flows[datapath_id] = set()

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
            # Operational live definition:
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
            # Build 19-feature record
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
                "pktrate": packet_rate,

                # 12
                "Pairflow": pairflow,

                # 13
                "Protocol": protocol,

                # 14
                "port_no": in_port,

                # 15-19
                # These will be populated from
                # port statistics by Ryu.
                "tx_bytes": 0,

                "rx_bytes": 0,

                "tx_kbps": 0.0,

                "rx_kbps": 0.0,

                "tot_kbps": 0.0,

                # Additional monitoring fields
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

        return records