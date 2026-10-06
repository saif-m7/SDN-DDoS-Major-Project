import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Gauge,
  Network,
  Radio,
} from "lucide-react";

import {
  getTrafficAnalytics,
} from "../services/api";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

function Analytics() {
  const [records, setRecords] = useState([]);
  const [error, setError] = useState("");

  const loadAnalytics = async () => {
    try {
      const result = await getTrafficAnalytics();

      setRecords(
        result.records ||
        result.data ||
        []
      );

      setError("");
    } catch (err) {
      console.error(err);
      setError("Unable to load traffic analytics.");
    }
  };

  useEffect(() => {
    loadAnalytics();

    const interval = setInterval(loadAnalytics, 5000);

    return () => clearInterval(interval);
  }, []);

  const analytics = useMemo(() => {
    if (!records.length) {
      return {
        averageBandwidth: 0,
        peakBandwidth: 0,
        totalBandwidth: 0,
        activePorts: 0,
      };
    }

    const bandwidthValues = records.map(
      (item) => Number(item.total_kbps || 0)
    );

    const averageBandwidth =
      bandwidthValues.reduce(
        (sum, value) => sum + value,
        0
      ) / bandwidthValues.length;

    const peakBandwidth = Math.max(
      ...bandwidthValues
    );

    const totalBandwidth =
      bandwidthValues.reduce(
        (sum, value) => sum + value,
        0
      );

    const activePorts = new Set(
      records.map(
        (item) =>
          `${item.switch}-${item.port}`
      )
    ).size;

    return {
      averageBandwidth,
      peakBandwidth,
      totalBandwidth,
      activePorts,
    };
  }, [records]);

  const chartData = records
    .slice()
    .reverse()
    .map((item, index) => ({
      sample: index + 1,
      bandwidth: Number(
        item.total_kbps || 0
      ),
    }));

  const portData = useMemo(() => {
    const grouped = {};

    records.forEach((item) => {
      const port =
        `S${item.switch || "?"}-P${item.port || "?"}`;

      if (!grouped[port]) {
        grouped[port] = {
          port,
          bandwidth: 0,
        };
      }

      grouped[port].bandwidth += Number(
        item.total_kbps || 0
      );
    });

    return Object.values(grouped)
      .sort(
        (a, b) =>
          b.bandwidth - a.bandwidth
      )
      .slice(0, 8);
  }, [records]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-400">
            Network Intelligence
          </p>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
            Traffic Analytics
          </h1>

          <p className="mt-1 text-sm text-zinc-600">
            Network traffic, bandwidth and port activity analysis
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-cyan-400/10 bg-cyan-400/[0.03] px-3 py-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-50" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-400" />
          </span>

          <span className="text-xs text-cyan-400">
            ANALYTICS ONLINE
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
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {/* Average */}
        <div className="rounded-xl border border-cyan-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Average Bandwidth
            </p>

            <Gauge className="h-4 w-4 text-cyan-400" />
          </div>

          <p className="mt-4 text-2xl font-semibold text-zinc-100">
            {analytics.averageBandwidth.toFixed(2)}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Kbps
          </p>
        </div>

        {/* Peak */}
        <div className="rounded-xl border border-amber-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Peak Bandwidth
            </p>

            <BarChart3 className="h-4 w-4 text-amber-400" />
          </div>

          <p className="mt-4 text-2xl font-semibold text-zinc-100">
            {analytics.peakBandwidth.toFixed(2)}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Kbps
          </p>
        </div>

        {/* Total */}
        <div className="rounded-xl border border-emerald-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Total Activity
            </p>

            <Activity className="h-4 w-4 text-emerald-400" />
          </div>

          <p className="mt-4 text-2xl font-semibold text-zinc-100">
            {analytics.totalBandwidth.toFixed(2)}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Kbps samples
          </p>
        </div>

        {/* Ports */}
        <div className="rounded-xl border border-violet-400/10 bg-[#0b0e13] p-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
              Active Ports
            </p>

            <Network className="h-4 w-4 text-violet-400" />
          </div>

          <p className="mt-4 text-2xl font-semibold text-zinc-100">
            {analytics.activePorts}
          </p>

          <p className="mt-1 text-xs text-zinc-600">
            Observed switch ports
          </p>
        </div>
      </div>

      {/* Bandwidth chart */}
      <div className="rounded-xl border border-white/[0.06] bg-[#0b0e13] p-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />

              <h2 className="text-sm font-medium text-zinc-200">
                Bandwidth Activity
              </h2>
            </div>

            <p className="mt-1 text-xs text-zinc-600">
              Network bandwidth across collected traffic samples
            </p>
          </div>

          <span className="rounded-md border border-white/[0.06] px-2 py-1 text-[10px] text-zinc-500">
            LIVE DATA
          </span>
        </div>

        <div className="mt-6 h-[330px]">
          <ResponsiveContainer
            width="100%"
            height="100%"
          >
            <AreaChart
              data={chartData}
              margin={{
                top: 10,
                right: 10,
                left: -20,
                bottom: 0,
              }}
            >
              <defs>
                <linearGradient
                  id="analyticsFill"
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="0%"
                    stopColor="#22d3ee"
                    stopOpacity={0.22}
                  />

                  <stop
                    offset="100%"
                    stopColor="#22d3ee"
                    stopOpacity={0}
                  />
                </linearGradient>
              </defs>

              <CartesianGrid
                stroke="rgba(255,255,255,0.05)"
                vertical={false}
              />

              <XAxis
                dataKey="sample"
                tick={{
                  fill: "#52525b",
                  fontSize: 10,
                }}
                axisLine={false}
                tickLine={false}
              />

              <YAxis
                tick={{
                  fill: "#52525b",
                  fontSize: 10,
                }}
                axisLine={false}
                tickLine={false}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: "#0b0e13",
                  border:
                    "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "8px",
                  color: "#e4e4e7",
                  fontSize: "12px",
                }}
                formatter={(value) => [
                  `${Number(value).toFixed(2)} Kbps`,
                  "Bandwidth",
                ]}
                labelFormatter={(label) =>
                  `Sample ${label}`
                }
              />

              <Area
                type="monotone"
                dataKey="bandwidth"
                stroke="#22d3ee"
                strokeWidth={2}
                fill="url(#analyticsFill)"
                dot={false}
                activeDot={{
                  r: 4,
                  fill: "#22d3ee",
                  stroke: "#07090d",
                  strokeWidth: 2,
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Port activity */}
      <div className="rounded-xl border border-white/[0.06] bg-[#0b0e13] p-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Radio className="h-4 w-4 text-violet-400" />

              <h2 className="text-sm font-medium text-zinc-200">
                Port Activity
              </h2>
            </div>

            <p className="mt-1 text-xs text-zinc-600">
              Bandwidth distribution across observed SDN ports
            </p>
          </div>
        </div>

        <div className="mt-6 h-[300px]">
          <ResponsiveContainer
            width="100%"
            height="100%"
          >
            <BarChart
              data={portData}
              margin={{
                top: 10,
                right: 10,
                left: -20,
                bottom: 0,
              }}
            >
              <CartesianGrid
                stroke="rgba(255,255,255,0.05)"
                vertical={false}
              />

              <XAxis
                dataKey="port"
                tick={{
                  fill: "#52525b",
                  fontSize: 10,
                }}
                axisLine={false}
                tickLine={false}
              />

              <YAxis
                tick={{
                  fill: "#52525b",
                  fontSize: 10,
                }}
                axisLine={false}
                tickLine={false}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: "#0b0e13",
                  border:
                    "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "8px",
                  color: "#e4e4e7",
                  fontSize: "12px",
                }}
                formatter={(value) => [
                  `${Number(value).toFixed(2)} Kbps`,
                  "Activity",
                ]}
              />

              <Bar
                dataKey="bandwidth"
                fill="#a78bfa"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default Analytics;