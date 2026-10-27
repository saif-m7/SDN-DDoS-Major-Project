
from mininet.topo import Topo


class CustomSDNTopo(Topo):
    """
    Three-switch SDN topology for DDoS detection
    and mitigation.

    S1: H1-H3 and simulated external attacker
    S2: H4-H6
    S3: H7-H9

    All switches are interconnected.
    """

    def build(self, link_bw=10, host_bw=100):

        # 1. Create OpenFlow 1.3 switches
        s1 = self.addSwitch("s1", protocols="OpenFlow13")
        s2 = self.addSwitch("s2", protocols="OpenFlow13")
        s3 = self.addSwitch("s3", protocols="OpenFlow13")

        # 2. Connect legitimate hosts to S1
        for i in range(1, 4):
            host = self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24"
            )
            self.addLink(host, s1, bw=host_bw)

        # 3. Connect legitimate hosts to S2
        for i in range(4, 7):
            host = self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24"
            )
            self.addLink(host, s2, bw=host_bw)

        # 4. Connect legitimate hosts to S3
        for i in range(7, 10):
            host = self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24"
            )
            self.addLink(host, s3, bw=host_bw)

        # 5. Simulated external attacker on S1
        attacker = self.addHost(
            "attacker",
            ip="10.0.0.10/24"
        )
        self.addLink(attacker, s1, bw=host_bw)

        # 6. Interconnect the switches
        self.addLink(s1, s2, bw=link_bw)
        self.addLink(s2, s3, bw=link_bw)


topos = {
    "customtopo": lambda: CustomSDNTopo(
        link_bw=10,
        host_bw=100
    )
}

