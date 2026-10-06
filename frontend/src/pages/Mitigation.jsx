import { useEffect, useMemo, useState } from "react";
import {
  ShieldCheck,
  ShieldAlert,
  Ban,
  Activity,
  Radio,
} from "lucide-react";

import api from "../services/api";

function Mitigation() {
  const [records, setRecords] = useState([]);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  const loadMitigation = async () => {
    try {
      const response = await api.get("/mitigation/status");

      const data = response.data;

      setRecords(
        data.records ||
        data.data ||
        []
      );

      setSummary(data.summary || null);
      setError("");
    } catch (err) {
      console.error(err);
      setError("Unable to load mitigation data.");
    }
  };

  useEffect(() => {
    loadMitigation();

    const interval = setInterval(
      loadMitigation,
      5000
    );

    return () => clearInterval(interval);
  }, []);

  const maliciousRecords = useMemo(
    () =>
      records.filter(
        (item) =>
          String(
            item.attack_status ||
            item.d1_prediction ||
            item.prediction ||
            ""
          ).toUpperCase() === "MALICIOUS"
      ),
    [records]
  );

  const mitigatedRecords = useMemo(
    () =>
      records.filter(
        (item) =>
          String(
            item.mitigation_action ||
            item.mitigation_status ||
            ""
          ).toUpperCase() !== "NONE"
      ),
    [records]
  );

  const mitigationRate =
    summary?.mitigation_rate ??
    (records.length
      ? (mitigatedRecords.length /
          records.length) *
        100
      : 0);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-400">
            SDN Response Layer
          </p>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
            DDoS Mitigation
          </h1>

          <p className="mt-1 text-sm text-zinc-600">
            Attack response and mitigation activity
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-emerald-400/10 bg-emerald-400/[0.03] px-3 py-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
          </span>

          <span className="text-xs text-emerald-400">
            MITIGATION ACTIVE
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
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        {/* Malicious */}
        <Stat
          label="Malicious Samples"
          value={
            summary?.malicious_samples ??
            maliciousRecords.length
          }
          description="Detected threats"
          icon={ShieldAlert}
          color="red"
        />

        {/* Mitigation */}
        <Stat
          label="Mitigation Required"
          value={
            summary?.mitigation_needed ??
            mitigatedRecords.length
          }
          description="Threats requiring response"
          icon={Ban}
          color="amber"
        />

        {/* Rate */}
        <Stat
          label="Mitigation Rate"
          value={`${Number(
            mitigationRate
          ).toFixed(1)}%`}
          description="Successful response rate"
          icon={ShieldCheck}
          color="emerald"
        />

        {/* Total */}
        <Stat
          label="Total Samples"
          value={
            summary?.total_samples ??
            records.length
          }
          description="Analyzed traffic"
          icon={Activity}
          color="cyan"
        />
      </div>

      {/* Protection status */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 rounded-xl border border-emerald-400/10 bg-[#0b0e13] p-6">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-emerald-400" />

                <h2 className="text-sm font-medium text-zinc-200">
                  SDN Protection Status
                </h2>
              </div>

              <p className="mt-1 text-xs text-zinc-600">
                Centralized controller response layer
              </p>
            </div>

            <span className="rounded-md border border-emerald-400/10 bg-emerald-400/[0.04] px-2.5 py-1 text-[10px] text-emerald-400">
              ACTIVE
            </span>
          </div>

          <div className="mt-8 flex items-center gap-6">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-emerald-400/10 bg-emerald-400/[0.05]">
              <ShieldCheck className="h-8 w-8 text-emerald-400" />
            </div>

            <div>
              <p className="text-lg font-semibold text-zinc-100">
                Network Protected
              </p>

              <p className="mt-1 text-sm text-zinc-600">
                Malicious traffic is identified and blocked through the SDN controller.
              </p>
            </div>
          </div>

          <div className="mt-8 grid grid-cols-2 gap-4">
            <div className="rounded-lg border border-white/[0.05] bg-white/[0.015] p-4">
              <p className="text-[10px] uppercase tracking-[0.12em] text-zinc-600">
                DDoS Detection
              </p>

              <p className="mt-2 text-sm font-medium text-emerald-400">
                Operational
              </p>
            </div>

            <div className="rounded-lg border border-white/[0.05] bg-white/[0.015] p-4">
              <p className="text-[10px] uppercase tracking-[0.12em] text-zinc-600">
                SDN Mitigation
              </p>

              <p className="mt-2 text-sm font-medium text-emerald-400">
                Operational
              </p>
            </div>
          </div>
        </div>

        {/* Response summary */}
        <div className="rounded-xl border border-white/[0.06] bg-[#0b0e13] p-6">
          <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
            Response Summary
          </p>

          <div className="mt-6 space-y-5">
            <SummaryRow
              label="Normal samples"
              value={
                summary?.normal_samples ??
                "—"
              }
            />

            <SummaryRow
              label="Malicious samples"
              value={
                summary?.malicious_samples ??
                maliciousRecords.length
              }
              danger
            />

            <SummaryRow
              label="Mitigation required"
              value={
                summary?.mitigation_needed ??
                mitigatedRecords.length
              }
              danger
            />

            <SummaryRow
              label="Mitigation rate"
              value={`${Number(
                mitigationRate
              ).toFixed(1)}%`}
              success
            />
          </div>

          <div className="mt-8 border-t border-white/[0.06] pt-5">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
              </span>

              <span className="text-xs text-zinc-500">
                Controller protection layer online
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mitigation history */}
      <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-[#0b0e13]">
        <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-5">
          <div>
            <div className="flex items-center gap-2">
              <Radio className="h-4 w-4 text-cyan-400" />

              <h2 className="text-sm font-medium text-zinc-200">
                Mitigation Activity
              </h2>
            </div>

            <p className="mt-1 text-xs text-zinc-600">
              Attack response actions recorded by the security system
            </p>
          </div>

          <span className="text-[10px] text-zinc-500">
            LIVE
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[1000px] text-left">
            <thead>
              <tr className="border-b border-white/[0.06] text-[10px] uppercase tracking-[0.12em] text-zinc-600">
                <th className="px-6 py-4 font-medium">
                  Timestamp
                </th>

                <th className="px-4 py-4 font-medium">
                  Status
                </th>

                <th className="px-4 py-4 font-medium">
                  Confidence
                </th>

                <th className="px-4 py-4 font-medium">
                  Attack Type
                </th>

                <th className="px-4 py-4 font-medium">
                  Action
                </th>

                <th className="px-6 py-4 font-medium">
                  Result
                </th>
              </tr>
            </thead>

            <tbody>
              {records.map((item, index) => {
                const status = String(
                  item.attack_status ||
                  item.d1_prediction ||
                  item.prediction ||
                  ""
                ).toUpperCase();

                const action =
                  item.mitigation_action ||
                  item.mitigation_status ||
                  item.mitigation ||
                  "NONE";

                const confidence = Number(
                  item.confidence ??
                  item.malicious_probability ??
                  0
                );

                const isMitigated =
                  String(action).toUpperCase() !==
                  "NONE";

                return (
                  <tr
                    key={`${item.timestamp}-${index}`}
                    className="border-b border-white/[0.04] transition hover:bg-white/[0.02]"
                  >
                    <td className="whitespace-nowrap px-6 py-4 text-xs text-zinc-500">
                      {item.timestamp || "—"}
                    </td>

                    <td className="px-4 py-4">
                      <span className="inline-flex items-center gap-2 rounded-md border border-red-400/10 bg-red-400/[0.05] px-2.5 py-1 text-[11px] text-red-400">
                        <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
                        {status || "MALICIOUS"}
                      </span>
                    </td>

                    <td className="px-4 py-4 text-xs text-zinc-400">
                      {(confidence * 100).toFixed(1)}%
                    </td>

                    <td className="px-4 py-4 text-xs text-zinc-400">
                      {item.predicted_class ||
                        item.attack_type ||
                        "DDoS"}
                    </td>

                    <td className="px-4 py-4">
                      <span className="text-xs font-medium text-amber-400">
                        {action}
                      </span>
                    </td>

                    <td className="px-6 py-4">
                      {isMitigated ? (
                        <span className="inline-flex items-center gap-2 text-xs text-emerald-400">
                          <ShieldCheck className="h-3.5 w-3.5" />
                          BLOCKED
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

              {records.length === 0 && !error && (
                <tr>
                  <td
                    colSpan="6"
                    className="px-6 py-16 text-center text-sm text-zinc-600"
                  >
                    Waiting for mitigation telemetry...
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

function Stat({
  label,
  value,
  description,
  icon: Icon,
  color,
}) {
  const colors = {
    red: "text-red-400 border-red-400/10",
    amber: "text-amber-400 border-amber-400/10",
    emerald: "text-emerald-400 border-emerald-400/10",
    cyan: "text-cyan-400 border-cyan-400/10",
  };

  return (
    <div
      className={`rounded-xl border bg-[#0b0e13] p-5 ${colors[color]}`}
    >
      <div className="flex items-center justify-between">
        <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
          {label}
        </p>

        <Icon className={`h-4 w-4 ${colors[color].split(" ")[0]}`} />
      </div>

      <p className="mt-4 text-2xl font-semibold text-zinc-100">
        {typeof value === "number"
          ? value.toLocaleString()
          : value}
      </p>

      <p className="mt-1 text-xs text-zinc-600">
        {description}
      </p>
    </div>
  );
}

function SummaryRow({
  label,
  value,
  danger,
  success,
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-zinc-600">
        {label}
      </span>

      <span
        className={
          danger
            ? "text-sm text-red-400"
            : success
            ? "text-sm text-emerald-400"
            : "text-sm text-zinc-300"
        }
      >
        {typeof value === "number"
          ? value.toLocaleString()
          : value}
      </span>
    </div>
  );
}

export default Mitigation;