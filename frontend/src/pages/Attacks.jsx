import { useEffect, useState } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Radio,
  Activity,
} from "lucide-react";

import api from "../services/api";

function Attacks() {
  const [attacks, setAttacks] = useState([]);
  const [error, setError] = useState("");

  const loadAttacks = async () => {
    try {
      const response = await api.get("/attacks");

      setAttacks(
        response.data.records ||
        response.data.data ||
        []
      );

      setError("");
    } catch (err) {
      console.error(err);
      setError("Unable to load attack records.");
    }
  };

  useEffect(() => {
    loadAttacks();

    const interval = setInterval(loadAttacks, 5000);

    return () => clearInterval(interval);
  }, []);

  const mitigated = attacks.filter(
    (item) =>
      String(
        item.mitigation_status ||
        item.mitigation ||
        ""
      ).toUpperCase() === "MITIGATED"
  );

  const averageConfidence =
    attacks.length > 0
      ? attacks.reduce(
          (sum, item) =>
            sum +
            Number(
              item.malicious_probability ||
              item.confidence ||
              0
            ),
          0
        ) / attacks.length
      : 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-red-400">
            Threat Intelligence
          </p>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
            Attack Monitor
          </h1>

          <p className="mt-1 text-sm text-zinc-600">
            Detected DDoS attacks and SDN mitigation activity
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-red-400/10 bg-red-400/[0.03] px-3 py-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-50" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-red-400" />
          </span>

          <span className="text-xs text-red-400">
            THREAT MONITORING
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

      {/* Statistics */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {/* Total attacks */}
        <div className="rounded-xl border border-red-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Detected Attacks
            </p>

            <ShieldAlert className="h-4 w-4 text-red-400" />
          </div>

          <p className="mt-4 text-3xl font-semibold text-red-400">
            {attacks.length}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            DDoS threat records
          </p>
        </div>

        {/* Mitigated */}
        <div className="rounded-xl border border-cyan-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Mitigated
            </p>

            <ShieldCheck className="h-4 w-4 text-cyan-400" />
          </div>

          <p className="mt-4 text-3xl font-semibold text-cyan-400">
            {mitigated.length}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Attacks blocked by SDN
          </p>
        </div>

        {/* Confidence */}
        <div className="rounded-xl border border-amber-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Avg. Confidence
            </p>

            <Activity className="h-4 w-4 text-amber-400" />
          </div>

          <p className="mt-4 text-3xl font-semibold text-amber-400">
            {(averageConfidence * 100).toFixed(1)}%
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            D1 malicious confidence
          </p>
        </div>
      </div>

      {/* Attack feed */}
      <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-[#0b0e13]">
        <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-5">
          <div>
            <div className="flex items-center gap-2">
              <Radio className="h-4 w-4 text-red-400" />

              <h2 className="text-sm font-medium text-zinc-200">
                Attack Feed
              </h2>
            </div>

            <p className="mt-1 text-xs text-zinc-600">
              Malicious traffic identified by the D1 detection engine
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-red-400" />

            <span className="text-[10px] text-zinc-500">
              LIVE
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[1150px] text-left">
            <thead>
              <tr className="border-b border-white/[0.06] text-[10px] uppercase tracking-[0.12em] text-zinc-600">
                <th className="px-6 py-4 font-medium">
                  Timestamp
                </th>

                <th className="px-4 py-4 font-medium">
                  Source
                </th>

                <th className="px-4 py-4 font-medium">
                  Destination
                </th>

                <th className="px-4 py-4 font-medium">
                  Protocol
                </th>

                <th className="px-4 py-4 font-medium">
                  Prediction
                </th>

                <th className="px-4 py-4 font-medium">
                  Confidence
                </th>

                <th className="px-6 py-4 font-medium">
                  Mitigation
                </th>
              </tr>
            </thead>

            <tbody>
              {attacks.map((item, index) => {
                const confidence = Number(
                  item.malicious_probability ||
                  item.confidence ||
                  0
                );

                const mitigation =
                  String(
                    item.mitigation_status ||
                    item.mitigation ||
                    ""
                  ).toUpperCase();

                return (
                  <tr
                    key={`${item.timestamp}-${index}`}
                    className="border-b border-white/[0.04] transition hover:bg-white/[0.02]"
                  >
                    <td className="whitespace-nowrap px-6 py-4 text-xs text-zinc-500">
                      {item.timestamp || "—"}
                    </td>

                    <td className="px-4 py-4 font-mono text-xs text-zinc-300">
                      {item.ipv4_src ||
                        item.src_ip ||
                        "—"}
                    </td>

                    <td className="px-4 py-4 font-mono text-xs text-zinc-300">
                      {item.ipv4_dst ||
                        item.dst_ip ||
                        "—"}
                    </td>

                    <td className="px-4 py-4">
                      <span className="rounded-md border border-white/[0.06] bg-white/[0.02] px-2 py-1 text-xs text-zinc-400">
                        {item.ip_proto ?? "—"}
                      </span>
                    </td>

                    <td className="px-4 py-4">
                      <span className="inline-flex items-center gap-2 rounded-md border border-red-400/10 bg-red-400/[0.05] px-2.5 py-1 text-[11px] font-medium text-red-400">
                        <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
                        {item.d1_prediction ||
                          item.prediction ||
                          "MALICIOUS"}
                      </span>
                    </td>

                    <td className="px-4 py-4">
                      <div className="flex items-center gap-3">
                        <div className="h-1.5 w-20 overflow-hidden rounded-full bg-white/[0.06]">
                          <div
                            className="h-full rounded-full bg-red-400"
                            style={{
                              width: `${Math.min(
                                confidence * 100,
                                100
                              )}%`,
                            }}
                          />
                        </div>

                        <span className="text-xs text-zinc-400">
                          {(confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>

                    <td className="px-6 py-4">
                      {mitigation === "MITIGATED" ? (
                        <span className="inline-flex items-center gap-2 text-xs text-cyan-400">
                          <ShieldCheck className="h-3.5 w-3.5" />
                          MITIGATED
                        </span>
                      ) : (
                        <span className="text-xs text-zinc-600">
                          NONE
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}

              {attacks.length === 0 && !error && (
                <tr>
                  <td
                    colSpan="7"
                    className="px-6 py-16 text-center text-sm text-zinc-600"
                  >
                    No attack records available.
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

export default Attacks;