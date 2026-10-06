import { useEffect, useState } from "react";
import {
  Activity,
  Radio,
  ArrowDownToLine,
  ArrowUpFromLine,
} from "lucide-react";

import { getLiveTraffic } from "../services/api";

function LiveTraffic() {
  const [traffic, setTraffic] = useState([]);
  const [error, setError] = useState("");

  const loadTraffic = async () => {
    try {
      const result = await getLiveTraffic();

      setTraffic(
        result.records || result.data || []
      );

      setError("");
    } catch (err) {
      console.error(err);
      setError("Unable to load live traffic data.");
    }
  };

  useEffect(() => {
    loadTraffic();

    const interval = setInterval(loadTraffic, 5000);

    return () => clearInterval(interval);
  }, []);

  const totalRx = traffic.reduce(
    (sum, item) => sum + Number(item.rx_bytes || 0),
    0
  );

  const totalTx = traffic.reduce(
    (sum, item) => sum + Number(item.tx_bytes || 0),
    0
  );

  const totalBandwidth = traffic.reduce(
    (sum, item) => sum + Number(item.total_kbps || 0),
    0
  );

  return (
    <div className="space-y-8">

      {/* Header */}
      <div className="flex items-end justify-between">

        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-400">
            Network Monitoring
          </p>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
            Live Traffic
          </h1>

          <p className="mt-1 text-sm text-zinc-600">
            Real-time SDN flow and bandwidth telemetry
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-emerald-400/10 bg-emerald-400/[0.03] px-3 py-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
          </span>

          <span className="text-xs text-emerald-400">
            LIVE
          </span>
        </div>

      </div>

      {/* Error */}
      {error && (
        <div className="rounded-xl border border-red-400/10 bg-red-400/[0.03] p-4">
          <p className="text-sm text-red-400">
            {error}
          </p>
        </div>
      )}

      {/* Summary */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">

        <div className="rounded-xl border border-white/[0.06] bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              RX Bytes
            </p>

            <ArrowDownToLine className="h-4 w-4 text-cyan-400" />
          </div>

          <p className="mt-4 text-2xl font-semibold text-zinc-100">
            {totalRx.toLocaleString()}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Received across monitored ports
          </p>
        </div>

        <div className="rounded-xl border border-white/[0.06] bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              TX Bytes
            </p>

            <ArrowUpFromLine className="h-4 w-4 text-cyan-400" />
          </div>

          <p className="mt-4 text-2xl font-semibold text-zinc-100">
            {totalTx.toLocaleString()}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Transmitted across monitored ports
          </p>
        </div>

        <div className="rounded-xl border border-white/[0.06] bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Total Bandwidth
            </p>

            <Activity className="h-4 w-4 text-cyan-400" />
          </div>

          <p className="mt-4 text-2xl font-semibold text-zinc-100">
            {totalBandwidth.toFixed(2)}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Kbps across current records
          </p>
        </div>

      </div>

      {/* Traffic table */}
      <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-[#0b0e13]">

        <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-5">

          <div>
            <div className="flex items-center gap-2">

              <Radio className="h-4 w-4 text-cyan-400" />

              <h2 className="text-sm font-medium text-zinc-200">
                Network Telemetry
              </h2>

            </div>

            <p className="mt-1 text-xs text-zinc-600">
              Latest traffic statistics from the SDN controller
            </p>
          </div>

          <span className="rounded-md border border-white/[0.06] px-2 py-1 text-[10px] text-zinc-500">
            {traffic.length} RECORDS
          </span>

        </div>

        <div className="overflow-x-auto">

          <table className="w-full min-w-[900px] text-left">

            <thead>
              <tr className="border-b border-white/[0.06] text-[10px] uppercase tracking-[0.12em] text-zinc-600">

                <th className="px-6 py-4 font-medium">
                  Timestamp
                </th>

                <th className="px-4 py-4 font-medium">
                  Switch
                </th>

                <th className="px-4 py-4 font-medium">
                  Port
                </th>

                <th className="px-4 py-4 font-medium">
                  RX Bytes
                </th>

                <th className="px-4 py-4 font-medium">
                  TX Bytes
                </th>

                <th className="px-4 py-4 font-medium">
                  RX Packets
                </th>

                <th className="px-4 py-4 font-medium">
                  TX Packets
                </th>

                <th className="px-6 py-4 font-medium">
                  Bandwidth
                </th>

              </tr>
            </thead>

            <tbody>

              {traffic.map((item, index) => (
                <tr
                  key={`${item.timestamp}-${item.switch}-${item.port}-${index}`}
                  className="border-b border-white/[0.04] transition hover:bg-white/[0.02]"
                >

                  <td className="whitespace-nowrap px-6 py-4 text-xs text-zinc-500">
                    {item.timestamp || "—"}
                  </td>

                  <td className="px-4 py-4 text-xs text-zinc-300">
                    {item.switch ?? "—"}
                  </td>

                  <td className="px-4 py-4">
                    <span className="rounded-md border border-white/[0.06] bg-white/[0.02] px-2 py-1 text-xs text-zinc-400">
                      {item.port ?? "—"}
                    </span>
                  </td>

                  <td className="px-4 py-4 text-xs text-zinc-400">
                    {Number(item.rx_bytes || 0).toLocaleString()}
                  </td>

                  <td className="px-4 py-4 text-xs text-zinc-400">
                    {Number(item.tx_bytes || 0).toLocaleString()}
                  </td>

                  <td className="px-4 py-4 text-xs text-zinc-400">
                    {Number(item.rx_packets || 0).toLocaleString()}
                  </td>

                  <td className="px-4 py-4 text-xs text-zinc-400">
                    {Number(item.tx_packets || 0).toLocaleString()}
                  </td>

                  <td className="px-6 py-4">
                    <span className="text-xs font-medium text-cyan-400">
                      {Number(item.total_kbps || 0).toFixed(2)} Kbps
                    </span>
                  </td>

                </tr>
              ))}

              {traffic.length === 0 && !error && (
                <tr>
                  <td
                    colSpan="8"
                    className="px-6 py-16 text-center text-sm text-zinc-600"
                  >
                    Waiting for traffic telemetry...
                  </td>
                </tr>
              )}

            </tbody>

          </table>

        </div>

      </div>

    </div>
  );
}

export default LiveTraffic;