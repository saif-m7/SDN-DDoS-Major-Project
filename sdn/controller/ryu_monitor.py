from ryu.base import app_manager



from ryu.controller import ofp_event



from ryu.controller.handler import (







    CONFIG_DISPATCHER,



    MAIN_DISPATCHER,



    DEAD_DISPATCHER,



    set_ev_cls







)







from ryu.ofproto import ofproto_v1_3



from ryu.lib import hub



from ryu.lib.packet import packet



from ryu.lib.packet import ethernet



from ryu.lib.packet import ipv4



from ryu.lib.packet import tcp



from ryu.lib.packet import udp



from ryu.lib.packet import icmp















import sys



import csv
import os



import time
from datetime import datetime







# =========================================================



# PYTHON PATH



# =========================================================







SDN_DIR = os.path.abspath(



    os.path.join(os.path.dirname(__file__), "..")



)







if SDN_DIR not in sys.path:



    sys.path.insert(0, SDN_DIR)







from monitoring.flow_collector import FlowCollector



from controller import d1_detector



from controller.d2_detector import D2Detector



























# =========================================================







# SDN TRAFFIC MONITOR







# =========================================================















class SDNTrafficMonitor(app_manager.RyuApp):















    OFP_VERSIONS = [







        ofproto_v1_3.OFP_VERSION







    ]















    def __init__(self, *args, **kwargs):















        super(







            SDNTrafficMonitor,







            self







        ).__init__(*args, **kwargs)















        # -------------------------------------------------







        # Connected switches







        # -------------------------------------------------















        self.datapaths = {}















        # -------------------------------------------------







        # MAC learning table







        # -------------------------------------------------















        self.mac_to_port = {}















        # -------------------------------------------------







        # Flow statistics collector







        # -------------------------------------------------















        self.flow_collector = FlowCollector()



















        # -------------------------------------------------



        # D2 DDoS detector



        # -------------------------------------------------







        self.d2_detector = D2Detector()















        # -------------------------------------------------







        # Previous port statistics







        # -------------------------------------------------















        self.previous_ports = {}















        # -------------------------------------------------







        # Latest port features







        # -------------------------------------------------















        self.latest_port_features = {}















        # -------------------------------------------------







        # Latest detected protocol







        #







        # D1 encoding:







        # 0 = ICMP







        # 1 = TCP







        # 2 = UDP







        # -------------------------------------------------















        self.latest_protocol_by_port = {}

        self.latest_flow_identity = {}
        # -------------------------------------------------
        # Traffic & bandwidth analytics
        # -------------------------------------------------
        self.analytics_dir = os.path.join(SDN_DIR, "results", "analytics")
        self.analytics_file = os.path.join(
            self.analytics_dir,
            "traffic_bandwidth.csv"
        )
        os.makedirs(self.analytics_dir, exist_ok=True)

        self.analytics_headers = [
            "timestamp", "switch", "port",
            "rx_bytes", "tx_bytes", "rx_packets", "tx_packets",
            "rx_kbps", "tx_kbps", "total_kbps",
            "d1_prediction", "d1_class",
            "normal_probability", "malicious_probability",
            "mitigation_status",
            "ipv4_src", "ipv4_dst", "ip_proto",
            "src_port", "dst_port"
        ]

        if not os.path.exists(self.analytics_file):
            with open(self.analytics_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.analytics_headers)
                writer.writeheader()
















        # -------------------------------------------------







        # Start monitoring thread







        # -------------------------------------------------















        self.monitor_thread = hub.spawn(







            self._monitor







        )























    # =====================================================







    # SWITCH CONNECTION







    # =====================================================















    @set_ev_cls(







        ofp_event.EventOFPSwitchFeatures,







        CONFIG_DISPATCHER







    )







    def switch_features_handler(self, ev):















        datapath = ev.msg.datapath















        self.datapaths[







            datapath.id







        ] = datapath















        self.logger.info(







            "Switch connected: datapath_id=%s",







            datapath.id







        )















        # -------------------------------------------------







        # Clear old learned/protocol flows







        # -------------------------------------------------















        self._delete_non_table_miss_flows(







            datapath







        )















        # -------------------------------------------------







        # Reset MAC table







        # -------------------------------------------------















        self.mac_to_port[







            datapath.id







        ] = {}















        # -------------------------------------------------







        # Install table miss







        # -------------------------------------------------















        self._install_table_miss(







            datapath







        )























    # =====================================================







    # DELETE OLD FLOWS







    # =====================================================















    def _delete_non_table_miss_flows(self, datapath):















        ofproto = datapath.ofproto







        parser = datapath.ofproto_parser















        match = parser.OFPMatch()















        flow_mod = parser.OFPFlowMod(







            datapath=datapath,







            command=ofproto.OFPFC_DELETE,







            out_port=ofproto.OFPP_ANY,







            out_group=ofproto.OFPG_ANY,







            priority=1,







            match=match







        )















        datapath.send_msg(flow_mod)















        self.logger.info(







            "Cleared existing forwarding flows on switch %s",







            datapath.id







        )























    # =====================================================







    # TABLE-MISS FLOW







    # =====================================================















    def _install_table_miss(self, datapath):















        ofproto = datapath.ofproto







        parser = datapath.ofproto_parser















        match = parser.OFPMatch()















        actions = [







            parser.OFPActionOutput(







                ofproto.OFPP_CONTROLLER,







                ofproto.OFPCML_NO_BUFFER







            )







        ]















        self._add_flow(







            datapath,







            priority=0,







            match=match,







            actions=actions







        )























    # =====================================================







    # ADD FLOW







    # =====================================================















    def _add_flow(







        self,







        datapath,







        priority,







        match,







        actions







    ):















        ofproto = datapath.ofproto







        parser = datapath.ofproto_parser















        instructions = [







            parser.OFPInstructionActions(







                ofproto.OFPIT_APPLY_ACTIONS,







                actions







            )







        ]















        flow_mod = parser.OFPFlowMod(







            datapath=datapath,







            priority=priority,







            match=match,







            instructions=instructions







        )















        datapath.send_msg(







            flow_mod







        )























    # =====================================================







    # PACKET IN







    # =====================================================















    @set_ev_cls(







        ofp_event.EventOFPPacketIn,







        MAIN_DISPATCHER







    )







    def packet_in_handler(self, ev):















        msg = ev.msg







        datapath = msg.datapath















        ofproto = datapath.ofproto







        parser = datapath.ofproto_parser















        in_port = msg.match[







            "in_port"







        ]















        # -------------------------------------------------







        # Count PacketIn events







        # -------------------------------------------------















        self.flow_collector.increment_packetin(







            datapath.id







        )















        # -------------------------------------------------







        # Parse packet







        # -------------------------------------------------















        pkt = packet.Packet(







            msg.data







        )















        eth = pkt.get_protocol(







            ethernet.ethernet







        )















        if eth is None:







            return















        src = eth.src







        dst = eth.dst















        dpid = datapath.id















        # -------------------------------------------------







        # Detect protocol







        #







        # D1 encoding:







        # ICMP = 0







        # TCP  = 1







        # UDP  = 2







        # -------------------------------------------------















        protocol = None







        protocol_name = "UNKNOWN"















        # IMPORTANT:







        # Do not assume IPv4 exists simply because the packet







        # contains an IP-looking payload. We explicitly detect







        # IPv4 below based on EtherType.















        ipv4_packet = None







        tcp_packet = None







        udp_packet = None







        icmp_packet = None







        # -------------------------------------------------



        # D2 live feature extraction



        # -------------------------------------------------







        d2_features = None



















        # -------------------------------------------------







        # IPv4 protocol detection







        # -------------------------------------------------















        if eth.ethertype == 0x0800:















            ipv4_packet = pkt.get_protocol(







                ipv4.ipv4







            )















            if ipv4_packet is None:















                self.logger.warning(







                    "IPv4 EtherType detected but IPv4 "







                    "header could not be parsed: "







                    "switch=%s src=%s dst=%s in_port=%s",







                    dpid,







                    src,







                    dst,







                    in_port







                )















            else:















                ip_proto = ipv4_packet.proto















                # -----------------------------------------







                # ICMP







                # -----------------------------------------















                if ip_proto == 1:















                    icmp_packet = pkt.get_protocol(







                        icmp.icmp







                    )















                    if icmp_packet is not None:















                        protocol = 0







                        protocol_name = "ICMP"















                # -----------------------------------------







                # TCP







                # -----------------------------------------















                elif ip_proto == 6:















                    tcp_packet = pkt.get_protocol(







                        tcp.tcp







                    )















                    if tcp_packet is not None:















                        protocol = 1







                        protocol_name = "TCP"















                # -----------------------------------------







                # UDP







                # -----------------------------------------















                elif ip_proto == 17:















                    udp_packet = pkt.get_protocol(







                        udp.udp







                    )















                    if udp_packet is not None:















                        protocol = 2







                        protocol_name = "UDP"















        # -------------------------------------------------







        # Fallback protocol detection







        # -------------------------------------------------















        elif protocol is None:















            if pkt.get_protocol(







                icmp.icmp







            ) is not None:















                protocol = 0







                protocol_name = "ICMP"















            elif pkt.get_protocol(







                tcp.tcp







            ) is not None:















                protocol = 1







                protocol_name = "TCP"















            elif pkt.get_protocol(







                udp.udp







            ) is not None:















                protocol = 2







                protocol_name = "UDP"















        # -------------------------------------------------











        # -------------------------------------------------



        # D2 live feature extraction



        # -------------------------------------------------







        if ipv4_packet is not None:







            total_length = ipv4_packet.total_length



            ttl = ipv4_packet.ttl



            proto = ipv4_packet.proto



            csum = ipv4_packet.csum







            src_port = 0



            dst_port = 0



            tcp_flag = 0



            type_icmp = 0



            code_icmp = 0







            # TCP



            if tcp_packet is not None:



                src_port = tcp_packet.src_port



                dst_port = tcp_packet.dst_port



                tcp_flag = tcp_packet.bits







            # UDP



            elif udp_packet is not None:



                src_port = udp_packet.src_port



                dst_port = udp_packet.dst_port







            # ICMP



            elif icmp_packet is not None:



                type_icmp = icmp_packet.type



                code_icmp = icmp_packet.code







            # -------------------------------------------------



            # Calculate live tx_bytes_ave



            # -------------------------------------------------







            port_key = (



                dpid,



                in_port



            )







            port_features = self.latest_port_features.get(



                port_key



            )







            tx_bytes_ave = 0.0







            if port_features is not None:







                tx_bytes = port_features.get(



                    "tx_bytes",



                    0



                )







                tx_packets = port_features.get(



                    "tx_packets",



                    0



                )







                if tx_packets > 0:







                    tx_bytes_ave = (



                        tx_bytes / tx_packets



                    )







            # -------------------------------------------------



            # D2 feature dictionary



            # -------------------------------------------------







            d2_features = {



                "total_length": total_length,



                "ttl": ttl,



                "proto": proto,



                "csum": csum,



                "src_port": src_port,



                "dst_port": dst_port,



                "tcp_flag": tcp_flag,



                "type_icmp": type_icmp,



                "code_icmp": code_icmp,



                "tx_bytes_ave": tx_bytes_ave



            }







            self.logger.info(



                "D2 LIVE FEATURES | "



                "total_length=%s | "



                "ttl=%s | "



                "proto=%s | "



                "csum=%s | "



                "src_port=%s | "



                "dst_port=%s | "



                "tcp_flag=%s | "



                "type_icmp=%s | "



                "code_icmp=%s | "



                "tx_bytes_ave=%.3f",



                total_length,



                ttl,



                proto,



                csum,



                src_port,



                dst_port,



                tcp_flag,



                type_icmp,



                code_icmp,



                tx_bytes_ave



            )            # -------------------------------------------------



            # D2 live prediction



            # -------------------------------------------------







            try:







                d2_result = self.d2_detector.predict(



                    d2_features



                )







                self.logger.info(



                    "D2 PREDICTION | "



                    "label=%s | "



                    "status=%s | "



                    "confidence=%.4f",



                    d2_result["label"],



                    d2_result["status"],



                    d2_result["confidence"]



                )







            except Exception as e:







                self.logger.error(



                    "D2 prediction failed: %s",



                    e



                )















        # Store protocol by switch + port







        # -------------------------------------------------















        port_key = (







            dpid,







            in_port







        )















        if protocol is not None:















            self.latest_protocol_by_port[







                port_key







            ] = protocol

            # -------------------------------------------------
            # Store latest flow identity for D1 mitigation
            # -------------------------------------------------

            if ipv4_packet is not None:
                flow_identity = {
                    "ipv4_src": ipv4_packet.src,
                    "ipv4_dst": ipv4_packet.dst,
                    "ip_proto": ipv4_packet.proto,
                    "tcp_src": tcp_packet.src_port if tcp_packet is not None else None,
                    "tcp_dst": tcp_packet.dst_port if tcp_packet is not None else None,
                    "udp_src": udp_packet.src_port if udp_packet is not None else None,
                    "udp_dst": udp_packet.dst_port if udp_packet is not None else None,
                }

                self.latest_flow_identity[port_key] = flow_identity















        # -------------------------------------------------







        # MAC learning







        # -------------------------------------------------















        self.mac_to_port.setdefault(







            dpid,







            {}







        )















        self.mac_to_port[







            dpid







        ][src] = in_port















        # -------------------------------------------------







        # Packet log







        # -------------------------------------------------















        self.logger.info(







            "Packet: switch=%s src=%s dst=%s "







            "in_port=%s protocol=%s(%s)",







            dpid,







            src,







            dst,







            in_port,







            protocol,







            protocol_name







        )















        # -------------------------------------------------







        # Determine output port







        # -------------------------------------------------















        if dst in self.mac_to_port[







            dpid







        ]:















            out_port = self.mac_to_port[







                dpid







            ][dst]















        else:















            out_port = ofproto.OFPP_FLOOD















        actions = [







            parser.OFPActionOutput(







                out_port







            )







        ]















        # =================================================







        # INSTALL FORWARDING FLOW







        # =================================================















        if out_port != ofproto.OFPP_FLOOD:















            match = None















            # -------------------------------------------------







            # TCP







            # Priority = 20







            # -------------------------------------------------















            if (







                ipv4_packet is not None







                and protocol == 1







            ):















                tcp_packet = pkt.get_protocol(







                    tcp.tcp







                )















                if tcp_packet is not None:















                    match = parser.OFPMatch(







                        in_port=in_port,







                        eth_type=0x0800,







                        ipv4_src=ipv4_packet.src,







                        ipv4_dst=ipv4_packet.dst,







                        ip_proto=6,







                        tcp_src=tcp_packet.src_port,







                        tcp_dst=tcp_packet.dst_port







                    )















                else:















                    match = parser.OFPMatch(







                        in_port=in_port,







                        eth_type=0x0800,







                        ipv4_src=ipv4_packet.src,







                        ipv4_dst=ipv4_packet.dst,







                        ip_proto=6







                    )















            # -------------------------------------------------







            # UDP







            # Priority = 20







            # -------------------------------------------------















            elif (







                ipv4_packet is not None







                and protocol == 2







            ):















                udp_packet = pkt.get_protocol(







                    udp.udp







                )















                if udp_packet is not None:















                    match = parser.OFPMatch(







                        in_port=in_port,







                        eth_type=0x0800,







                        ipv4_src=ipv4_packet.src,







                        ipv4_dst=ipv4_packet.dst,







                        ip_proto=17,







                        udp_src=udp_packet.src_port,







                        udp_dst=udp_packet.dst_port







                    )















                else:















                    match = parser.OFPMatch(







                        in_port=in_port,







                        eth_type=0x0800,







                        ipv4_src=ipv4_packet.src,







                        ipv4_dst=ipv4_packet.dst,







                        ip_proto=17







                    )















            # -------------------------------------------------







            # ICMP







            # Priority = 20







            # -------------------------------------------------















            elif (







                ipv4_packet is not None







                and protocol == 0







            ):















                match = parser.OFPMatch(







                    in_port=in_port,







                    eth_type=0x0800,







                    ipv4_src=ipv4_packet.src,







                    ipv4_dst=ipv4_packet.dst,







                    ip_proto=1







                )















            # -------------------------------------------------







            # Other IPv4







            # Priority = 20







            # -------------------------------------------------















            elif ipv4_packet is not None:















                match = parser.OFPMatch(







                    in_port=in_port,







                    eth_type=0x0800,







                    ipv4_src=ipv4_packet.src,







                    ipv4_dst=ipv4_packet.dst,







                    ip_proto=ipv4_packet.proto







                )















            # -------------------------------------------------







            # Non-IPv4 fallback







            # Priority = 5







            # -------------------------------------------------















            else:















                match = parser.OFPMatch(







                    in_port=in_port,







                    eth_dst=dst,







                    eth_src=src







                )















            # =================================================







            # INSTALL CORRECT PRIORITY







            # =================================================















            if (







                ipv4_packet is not None







                and protocol in (0, 1, 2)







            ):















                # -------------------------------------------------







                # Protocol-aware IPv4 flow







                # Priority = 20







                #







                # TCP/UDP/ICMP traffic must use a protocol-aware







                # flow so that live traffic is correctly identified.







                # -------------------------------------------------















                self._add_flow(







                    datapath,







                    priority=20,







                    match=match,







                    actions=actions







                )















                self.logger.info(







                    "Installed protocol-aware flow: "







                    "switch=%s protocol=%s(%s) "







                    "in_port=%s out_port=%s",







                    dpid,







                    protocol,







                    protocol_name,







                    in_port,







                    out_port







                )















            elif (







                ipv4_packet is None







                and out_port != ofproto.OFPP_FLOOD







            ):















                # -------------------------------------------------







                # L2 fallback ONLY for non-IPv4 traffic







                # Priority = 5







                #







                # Do NOT install a broad L2 flow for IPv4 traffic.







                # Otherwise TCP/UDP traffic can be captured by the







                # generic L2 rule instead of a protocol-aware rule.







                # -------------------------------------------------















                self._add_flow(







                    datapath,







                    priority=5,







                    match=match,







                    actions=actions







                )















                self.logger.info(







                    "Installed L2 fallback flow: "







                    "switch=%s in_port=%s out_port=%s",







                    dpid,







                    in_port,







                    out_port







                )















            else:















                # -------------------------------------------------







                # IPv4 destination is not known yet.







                #







                # Do not install a broad L2 flow.







                # The current packet is forwarded using PacketOut.







                # Once the destination is learned, a subsequent







                # IPv4 packet can install a protocol-aware flow.







                # -------------------------------------------------















                self.logger.info(







                    "IPv4 destination unknown - "







                    "no broad L2 flow installed: "







                    "switch=%s protocol=%s(%s) "







                    "in_port=%s",







                    dpid,







                    protocol,







                    protocol_name,







                    in_port







                )















        # =================================================







        # SEND PACKET OUT







        # =================================================















        data = None















        if msg.buffer_id == ofproto.OFP_NO_BUFFER:















            data = msg.data















        out = parser.OFPPacketOut(







            datapath=datapath,







            buffer_id=msg.buffer_id,







            in_port=in_port,







            actions=actions,







            data=data







        )















        datapath.send_msg(







            out







        )























    # =====================================================







    # SWITCH STATE







    # =====================================================















    @set_ev_cls(







        ofp_event.EventOFPStateChange,







        [MAIN_DISPATCHER, DEAD_DISPATCHER]







    )







    def state_change_handler(self, ev):















        datapath = ev.datapath















        if ev.state == MAIN_DISPATCHER:















            self.datapaths[







                datapath.id







            ] = datapath















            self.logger.info(







                "Switch entered MAIN state: datapath_id=%s",







                datapath.id







            )















        elif ev.state == DEAD_DISPATCHER:















            if datapath.id in self.datapaths:















                del self.datapaths[







                    datapath.id







                ]















                self.logger.info(







                    "Switch disconnected: datapath_id=%s",







                    datapath.id







                )























    # =====================================================







    # MONITORING LOOP







    # =====================================================















    def _monitor(self):















        while True:















            for datapath in list(







                self.datapaths.values()







            ):















                # Flow statistics















                self._request_flow_stats(







                    datapath







                )















                # Port statistics















                self._request_port_stats(







                    datapath







                )















            # Every 5 seconds















            hub.sleep(5)























    # =====================================================







    # REQUEST FLOW STATISTICS







    # =====================================================















    def _request_flow_stats(self, datapath):















        parser = datapath.ofproto_parser















        request = parser.OFPFlowStatsRequest(







            datapath=datapath,







            match=parser.OFPMatch()







        )















        datapath.send_msg(







            request







        )























    # =====================================================







    # RECEIVE FLOW STATISTICS







    # =====================================================















    @set_ev_cls(







        ofp_event.EventOFPFlowStatsReply,







        MAIN_DISPATCHER







    )







    def flow_stats_reply_handler(self, ev):















        datapath = ev.msg.datapath















        self.logger.info(







            "===== FLOW STATISTICS | SWITCH %s =====",







            datapath.id







        )















        records = (







            self.flow_collector.process_flow_stats(







                datapath.id,







                ev.msg.body







            )







        )















        # -------------------------------------------------







        # Attach latest port features







        # -------------------------------------------------















        for record in records:















            port_key = (







                record["switch"],







                record["port_no"]







            )















            port_features = (







                self.latest_port_features.get(







                    port_key







                )







            )















            if port_features is not None:















                record["tx_bytes"] = (







                    port_features["tx_bytes"]







                )















                record["rx_bytes"] = (







                    port_features["rx_bytes"]







                )















                record["tx_kbps"] = (







                    port_features["tx_kbps"]







                )















                record["rx_kbps"] = (







                    port_features["rx_kbps"]







                )















                record["tot_kbps"] = (







                    port_features["tot_kbps"]







                )















            # -------------------------------------------------







            # Protocol fallback







            # -------------------------------------------------















            detected_protocol = (







                self.latest_protocol_by_port.get(







                    port_key







                )







            )















            if detected_protocol is not None:















                ip_proto = record.get(







                    "ip_proto"







                )















                # Only use PacketIn protocol







                # when flow stats did not provide







                # a recognized protocol.















                if ip_proto not in (







                    1,







                    6,







                    17







                ):















                    record["Protocol"] = (







                        detected_protocol







                    )















            # -------------------------------------------------







            # Display complete 19-feature record







            # -------------------------------------------------















            self.logger.info(







                "LIVE FEATURES | "







                "switch=%s | "







                "pktcount=%s | "







                "bytecount=%s | "







                "dur=%s | "







                "dur_nsec=%s | "







                "tot_dur=%s | "







                "flows=%s | "







                "packetins=%s | "







                "pktperflow=%.3f | "







                "byteperflow=%.3f | "







                "pktrate=%.3f | "







                "Pairflow=%s | "







                "Protocol=%s | "







                "port_no=%s | "







                "tx_bytes=%s | "







                "rx_bytes=%s | "







                "tx_kbps=%.3f | "







                "rx_kbps=%.3f | "







                "tot_kbps=%.3f",















                record["switch"],







                record["pktcount"],







                record["bytecount"],







                record["dur"],







                record["dur_nsec"],







                record["tot_dur"],







                record["flows"],







                record["packetins"],







                record["pktperflow"],







                record["byteperflow"],







                record["pktrate"],







                record["Pairflow"],







                record["Protocol"],







                record["port_no"],







                record["tx_bytes"],







                record["rx_bytes"],







                record["tx_kbps"],







                record["rx_kbps"],







                record["tot_kbps"]







                )# =====================================================



                # =====================================================

            # D1 DDoS DETECTION

            # =====================================================



            # -------------------------------------------------
            # Attach flow identity for D1 mitigation
            # -------------------------------------------------

            # Prefer identity already present in the flow-stat record.
            # Only fall back to the latest PacketIn identity when needed.
            port_key = (
                record["switch"],
                record["port_no"]
            )

            if record.get("ip_proto") not in (1, 6, 17):
                flow_identity = self.latest_flow_identity.get(port_key)
                if flow_identity is not None:
                    record.update(flow_identity)

            try:

                d1_result = d1_detector.detect_single_record(

                    record

                )



                self.logger.info(

                    "D1 PREDICTION | "

                    "prediction=%s | "

                    "class=%s | "

                    "normal_probability=%.4f | "

                    "malicious_probability=%.4f",

                    d1_result["prediction"],

                    d1_result["class"],

                    d1_result["normal_probability"],

                    d1_result["malicious_probability"]

                )
                # Store D1 detection event in analytics
                d1_mitigation_status = (
                    "MITIGATED" if d1_result["class"] == 1 else "NONE"
                )

                with open(
                    self.analytics_file,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=self.analytics_headers
                    )
                    writer.writerow({
                        "timestamp": datetime.now().isoformat(sep=" "),
                        "switch": record.get("switch"),
                        "port": record.get("port_no"),
                        "rx_bytes": record.get("rx_bytes"),
                        "tx_bytes": record.get("tx_bytes"),
                        "rx_packets": "",
                        "tx_packets": "",
                        "rx_kbps": record.get("rx_kbps"),
                        "tx_kbps": record.get("tx_kbps"),
                        "total_kbps": record.get("tot_kbps"),
                        "d1_prediction": d1_result["prediction"],
                        "d1_class": d1_result["class"],
                        "normal_probability": d1_result["normal_probability"],
                        "malicious_probability": d1_result["malicious_probability"],
                        "mitigation_status": d1_mitigation_status,
                        "ipv4_src": record.get("ipv4_src"),
                        "ipv4_dst": record.get("ipv4_dst"),
                        "ip_proto": record.get("ip_proto"),
                        "src_port": (
                            record.get("tcp_src")
                            if record.get("ip_proto") == 6
                            else record.get("udp_src")
                        ),
                        "dst_port": (
                            record.get("tcp_dst")
                            if record.get("ip_proto") == 6
                            else record.get("udp_dst")
                        )
                    })




                # =================================================

                # D1-TRIGGERED MITIGATION

                # =================================================



                if d1_result["class"] == 1:

                    self._mitigate_d1_attack(

                        datapath,

                        record

                    )



            except Exception as e:

                self.logger.error(

                    "D1 prediction failed: %s",

                    e

                )





    # =====================================================

    # D1 DDoS MITIGATION

    # =====================================================



    def _mitigate_d1_attack(self, datapath, record):



        ofproto = datapath.ofproto

        parser = datapath.ofproto_parser

        match_fields = {}



        if record.get("in_port") is not None:

            match_fields["in_port"] = record["in_port"]



        if record.get("ipv4_src") is not None:

            match_fields["eth_type"] = 0x0800

            match_fields["ipv4_src"] = record["ipv4_src"]



        if record.get("ipv4_dst") is not None:

            match_fields["ipv4_dst"] = record["ipv4_dst"]



        if record.get("ip_proto") is not None:

            match_fields["ip_proto"] = record["ip_proto"]



        if record.get("ip_proto") == 6:

            if record.get("tcp_src") is not None:

                match_fields["tcp_src"] = record["tcp_src"]

            if record.get("tcp_dst") is not None:

                match_fields["tcp_dst"] = record["tcp_dst"]



        elif record.get("ip_proto") == 17:

            if record.get("udp_src") is not None:

                match_fields["udp_src"] = record["udp_src"]

            if record.get("udp_dst") is not None:

                match_fields["udp_dst"] = record["udp_dst"]



        match = parser.OFPMatch(**match_fields)

        actions = []



        flow_mod = parser.OFPFlowMod(

            datapath=datapath,

            priority=200,

            match=match,

            instructions=[

                parser.OFPInstructionActions(

                    ofproto.OFPIT_APPLY_ACTIONS,

                    actions

                )

            ],

            idle_timeout=30,

            hard_timeout=60

        )



        datapath.send_msg(flow_mod)



        self.logger.warning(

            "D1 MITIGATION | DROP RULE INSTALLED | "

            "switch=%s | src=%s | dst=%s | protocol=%s | "

            "src_port=%s | dst_port=%s",

            datapath.id,

            record.get("ipv4_src"),

            record.get("ipv4_dst"),

            record.get("ip_proto"),

            record.get("tcp_src") if record.get("ip_proto") == 6 else record.get("udp_src"),

            record.get("tcp_dst") if record.get("ip_proto") == 6 else record.get("udp_dst")

        )





    # =====================================================

    # REQUEST PORT STATISTICS

    # =====================================================



    def _request_port_stats(self, datapath):















        ofproto = datapath.ofproto







        parser = datapath.ofproto_parser















        request = parser.OFPPortStatsRequest(







            datapath=datapath,







            port_no=ofproto.OFPP_ANY







        )















        datapath.send_msg(







            request







        )























    # =====================================================







    # RECEIVE PORT STATISTICS







    # =====================================================















    @set_ev_cls(







        ofp_event.EventOFPPortStatsReply,







        MAIN_DISPATCHER







    )







    def port_stats_reply_handler(self, ev):















        datapath = ev.msg.datapath















        current_time = time.time()















        self.logger.info(







            "===== PORT STATISTICS | SWITCH %s =====",







            datapath.id







        )















        for stat in ev.msg.body:















            # -------------------------------------------------







            # Ignore special OpenFlow ports







            # -------------------------------------------------















            if (







                stat.port_no >=







                datapath.ofproto.OFPP_MAX







            ):















                continue















            port_no = stat.port_no















            # -------------------------------------------------







            # Current cumulative counters







            # -------------------------------------------------















            rx_bytes = stat.rx_bytes







            tx_bytes = stat.tx_bytes















            rx_packets = stat.rx_packets







            tx_packets = stat.tx_packets















            # -------------------------------------------------







            # Port key







            # -------------------------------------------------















            port_key = (







                datapath.id,







                port_no







            )















            # -------------------------------------------------







            # First observation







            # -------------------------------------------------















            if port_key not in self.previous_ports:















                rx_byte_delta = 0







                tx_byte_delta = 0







                time_delta = 0.0















            # -------------------------------------------------







            # Existing port







            # -------------------------------------------------















            else:















                previous = self.previous_ports[







                    port_key







                ]















                rx_byte_delta = max(







                    0,







                    rx_bytes -







                    previous["rx_bytes"]







                )















                tx_byte_delta = max(







                    0,







                    tx_bytes -







                    previous["tx_bytes"]







                )















                time_delta = max(







                    0.0,







                    current_time -







                    previous["timestamp"]







                )















            # -------------------------------------------------







            # Calculate RX/TX rates







            # -------------------------------------------------















            if time_delta > 0:















                rx_byte_rate = (







                    rx_byte_delta /







                    time_delta







                )















                tx_byte_rate = (







                    tx_byte_delta /







                    time_delta







                )















            else:















                rx_byte_rate = 0.0







                tx_byte_rate = 0.0















            # -------------------------------------------------







            # Convert to kilobits/sec







            # -------------------------------------------------















            rx_kbps = (







                rx_byte_rate * 8 / 1000







            )















            tx_kbps = (







                tx_byte_rate * 8 / 1000







            )















            total_kbps = (







                rx_kbps +







                tx_kbps







            )















            # -------------------------------------------------







            # Store current values







            # -------------------------------------------------















            self.previous_ports[







                port_key







            ] = {















                "rx_bytes": rx_bytes,















                "tx_bytes": tx_bytes,















                "timestamp": current_time







            }















            # -------------------------------------------------







            # Store latest port features







            # -------------------------------------------------















            self.latest_port_features[







                port_key







            ] = {















                "rx_bytes": rx_bytes,















                "tx_bytes": tx_bytes,















                "rx_kbps": rx_kbps,















                "tx_kbps": tx_kbps,















                "tot_kbps": total_kbps,















                "rx_packets": rx_packets,















                "tx_packets": tx_packets,















                "timestamp": current_time







            }















            # -------------------------------------------------







            # Display port features







            # -------------------------------------------------















            self.logger.info(







                "PORT RECORD | "







                "switch=%s | "







                "port=%s | "







                "rx_bytes=%s | "







                "tx_bytes=%s | "







                "rx_packets=%s | "







                "tx_packets=%s | "







                "rx_kbps=%.3f | "







                "tx_kbps=%.3f | "







                "total_kbps=%.3f",















                datapath.id,















                port_no,















                rx_bytes,















                tx_bytes,















                rx_packets,















                tx_packets,















                rx_kbps,















                tx_kbps,















                total_kbps







            )
            # Store traffic & bandwidth analytics
            with open(
                self.analytics_file,
                "a",
                newline="",
                encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.analytics_headers
                )
                writer.writerow({
                    "timestamp": datetime.fromtimestamp(current_time).isoformat(sep=" "),
                    "switch": datapath.id,
                    "port": port_no,
                    "rx_bytes": rx_bytes,
                    "tx_bytes": tx_bytes,
                    "rx_packets": rx_packets,
                    "tx_packets": tx_packets,
                    "rx_kbps": rx_kbps,
                    "tx_kbps": tx_kbps,
                    "total_kbps": total_kbps,
                    "d1_prediction": "",
                    "d1_class": "",
                    "normal_probability": "",
                    "malicious_probability": "",
                    "mitigation_status": "",
                    "ipv4_src": "",
                    "ipv4_dst": "",
                    "ip_proto": "",
                    "src_port": "",
                    "dst_port": ""
                })

