import { useEffect, useState } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Target,
  Radio,
} from "lucide-react";

import { getLiveDetection } from "../services/api";

function Detection() {
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState("");

  const loadDetections = async () => {
    try {
      const result = await getLiveDetection();

      const records = result.records || result.data || [];

      // Keep only complete flow-level D1 detection records.
      // Port-only analytics records without flow identity are excluded.
      const flowDetections = records.filter(
        (item) =>
          item.ipv4_src &&
          item.ipv4_dst &&
          item.ip_proto !== null &&
          item.ip_proto !== undefined
      );

      setDetections(flowDetections);
      setError("");
    } catch (err) {
      console.error(err);
      setError("Unable to load detection data.");
    }
  };

  useEffect(() => {
    loadDetections();

    const interval = setInterval(loadDetections, 5000);

    return () => clearInterval(interval);
  }, []);

  const malicious = detections.filter(
    (item) =>
      String(item.d1_prediction || "").toUpperCase() === "MALICIOUS"
  );

  const normal = detections.filter(
    (item) =>
      String(item.d1_prediction || "").toUpperCase() === "NORMAL"
  );

  const mitigated = detections.filter(
    (item) =>
      String(item.mitigation_status || "").toUpperCase() === "MITIGATED"
  );

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-red-400">
            Threat Detection
          </p>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
            DDoS Detection
          </h1>

          <p className="mt-1 text-sm text-zinc-600">
            Real-time AI-based network threat identification
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-emerald-400/10 bg-emerald-400/[0.03] px-3 py-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
          </span>

          <span className="text-xs text-emerald-400">
            DETECTOR ONLINE
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

      {/* Security statistics */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {/* Malicious */}
        <div className="rounded-xl border border-red-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Malicious
            </p>

            <ShieldAlert className="h-4 w-4 text-red-400" />
          </div>

          <p className="mt-4 text-3xl font-semibold text-red-400">
            {malicious.length}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Threat records detected
          </p>
        </div>

        {/* Normal */}
        <div className="rounded-xl border border-emerald-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Normal
            </p>

            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          </div>

          <p className="mt-4 text-3xl font-semibold text-emerald-400">
            {normal.length}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Legitimate traffic records
          </p>
        </div>

        {/* Mitigated */}
        <div className="rounded-xl border border-cyan-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Mitigated
            </p>

            <Target className="h-4 w-4 text-cyan-400" />
          </div>

          <p className="mt-4 text-3xl font-semibold text-cyan-400">
            {mitigated.length}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Threats blocked by SDN
          </p>
        </div>
      </div>

      {/* Detection table */}
      <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-[#0b0e13]">
        <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-5">
          <div>
            <div className="flex items-center gap-2">
              <Radio className="h-4 w-4 text-red-400" />

              <h2 className="text-sm font-medium text-zinc-200">
                Live Detection Feed
              </h2>
            </div>

            <p className="mt-1 text-xs text-zinc-600">
              AI predictions generated by the D1 detection engine
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
          <table className="w-full min-w-[1100px] text-left">
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
                  Action
                </th>
              </tr>
            </thead>

            <tbody>
              {detections.map((item, index) => {
                const prediction = String(
                  item.d1_prediction || ""
                ).toUpperCase();

                const isMalicious =
                  prediction === "MALICIOUS";

                const confidence = isMalicious
                  ? Number(item.malicious_probability || 0)
                  : Number(item.normal_probability || 0);

                const mitigation =
                  String(
                    item.mitigation_status || ""
                  ).toUpperCase();

                return (
                  <tr
                    key={`${item.timestamp}-${index}`}
                    className="border-b border-white/[0.04] transition hover:bg-white/[0.02]"
                  >
                    {/* Timestamp */}
                    <td className="whitespace-nowrap px-6 py-4 text-xs text-zinc-500">
                      {item.timestamp || "—"}
                    </td>

                    {/* Source */}
                    <td className="px-4 py-4 text-xs font-mono text-zinc-300">
                      {item.ipv4_src || "—"}
                    </td>

                    {/* Destination */}
                    <td className="px-4 py-4 text-xs font-mono text-zinc-300">
                      {item.ipv4_dst || "—"}
                    </td>

                    {/* Protocol */}
                    <td className="px-4 py-4">
                      <span className="rounded-md border border-white/[0.06] bg-white/[0.02] px-2 py-1 text-xs text-zinc-400">
                        {item.ip_proto ?? "—"}
                      </span>
                    </td>

                    {/* Prediction */}
                    <td className="px-4 py-4">
                      <span
                        className={
                          isMalicious
                            ? "inline-flex items-center gap-2 rounded-md border border-red-400/10 bg-red-400/[0.05] px-2.5 py-1 text-[11px] font-medium text-red-400"
                            : "inline-flex items-center gap-2 rounded-md border border-emerald-400/10 bg-emerald-400/[0.05] px-2.5 py-1 text-[11px] font-medium text-emerald-400"
                        }
                      >
                        <span
                          className={
                            isMalicious
                              ? "h-1.5 w-1.5 rounded-full bg-red-400"
                              : "h-1.5 w-1.5 rounded-full bg-emerald-400"
                          }
                        />

                        {item.d1_prediction || "UNKNOWN"}
                      </span>
                    </td>

                    {/* Confidence */}
                    <td className="px-4 py-4">
                      <div className="flex items-center gap-3">
                        <div className="h-1.5 w-20 overflow-hidden rounded-full bg-white/[0.06]">
                          <div
                            className={
                              isMalicious
                                ? "h-full rounded-full bg-red-400"
                                : "h-full rounded-full bg-emerald-400"
                            }
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

                    {/* Action */}
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

              {detections.length === 0 && !error && (
                <tr>
                  <td
                    colSpan="7"
                    className="px-6 py-16 text-center text-sm text-zinc-600"
                  >
                    Waiting for detection telemetry...
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

export default Detection;