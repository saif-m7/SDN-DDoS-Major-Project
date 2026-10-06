import { useEffect, useState } from "react";
import {
  Activity,
  ShieldAlert,
  ShieldCheck,
  Network,
} from "lucide-react";

import StatCard from "../components/dashboard/StatCard";
import TrafficChart from "../components/dashboard/TrafficChart";

import {
  getDashboardOverview,
  getTrafficAnalytics,
} from "../services/api";

function Dashboard() {
  const [data, setData] = useState(null);
  const [trafficData, setTrafficData] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const [overview, traffic] = await Promise.all([
          getDashboardOverview(),
          getTrafficAnalytics(),
        ]);

        setData(overview);

        setTrafficData(
          traffic.records || traffic.data || []
        );
      } catch (err) {
        console.error(err);
        setError("Unable to connect to the security backend.");
      }
    };

    loadDashboard();
  }, []);

  if (error) {
    return (
      <div className="rounded-xl border border-red-400/10 bg-red-400/[0.03] p-6">
        <p className="text-sm text-red-400">{error}</p>

        <p className="mt-2 text-xs text-zinc-600">
          Make sure the Flask backend is running on port 5000.
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex min-h-[300px] items-center justify-center">
        <div className="flex items-center gap-3 text-sm text-zinc-500">
          <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-400" />

          Loading network intelligence...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">

      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-400">
            Security Operations
          </p>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
            Network Overview
          </h1>

          <p className="mt-1 text-sm text-zinc-600">
            Real-time SDN traffic and DDoS security intelligence
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-emerald-400/10 bg-emerald-400/[0.03] px-3 py-2">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />

          <span className="text-xs text-emerald-400">
            {data.network_status}
          </span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">

        <StatCard
          label="Total Traffic"
          value={Number(data.total_traffic).toLocaleString()}
          unit="KB"
          icon={Activity}
          accent="cyan"
        />

        <StatCard
          label="Normal Flows"
          value={Number(data.normal_flows).toLocaleString()}
          icon={Network}
          accent="emerald"
        />

        <StatCard
          label="Malicious Flows"
          value={Number(data.malicious_flows).toLocaleString()}
          icon={ShieldAlert}
          accent="red"
        />

        <StatCard
          label="Attacks Mitigated"
          value={Number(data.attacks_mitigated).toLocaleString()}
          icon={ShieldCheck}
          accent="amber"
        />

      </div>

      {/* Analytics */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">

        {/* Traffic Intelligence */}
        <div className="lg:col-span-2 rounded-xl border border-white/[0.06] bg-[#0b0e13] p-6">

          <div className="flex items-start justify-between">

            <div>
              <div className="flex items-center gap-2">

                <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />

                <h2 className="text-sm font-medium text-zinc-200">
                  Traffic Intelligence
                </h2>

              </div>

              <p className="mt-1 text-xs text-zinc-600">
                Network bandwidth activity
              </p>
            </div>

            <div className="rounded-md border border-white/[0.06] px-2 py-1">
              <span className="text-[10px] text-zinc-500">
                LIVE DATA
              </span>
            </div>

          </div>

          <div className="mt-6">
            <TrafficChart data={trafficData} />
          </div>

        </div>

        {/* Security Status */}
        <div className="rounded-xl border border-white/[0.06] bg-[#0b0e13] p-6">

          <p className="text-[10px] uppercase tracking-[0.15em] text-zinc-600">
            Security Status
          </p>

          <div className="mt-6 flex items-center gap-4">

            <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-emerald-400/10 bg-emerald-400/[0.05]">
              <ShieldCheck className="h-6 w-6 text-emerald-400" />
            </div>

            <div>
              <p className="text-sm font-medium text-zinc-200">
                Protection Active
              </p>

              <p className="mt-1 text-xs text-zinc-600">
                DDoS mitigation operational
              </p>
            </div>

          </div>

          <div className="mt-8 space-y-3">

            <div className="flex justify-between text-xs">
              <span className="text-zinc-600">
                Attacks detected
              </span>

              <span className="text-zinc-300">
                {Number(data.attacks_detected).toLocaleString()}
              </span>
            </div>

            <div className="flex justify-between text-xs">
              <span className="text-zinc-600">
                Attacks mitigated
              </span>

              <span className="text-emerald-400">
                {Number(data.attacks_mitigated).toLocaleString()}
              </span>
            </div>

            <div className="flex justify-between text-xs">
              <span className="text-zinc-600">
                Current bandwidth
              </span>

              <span className="text-zinc-300">
                {Number(data.current_bandwidth).toFixed(2)} Kbps
              </span>
            </div>

            <div className="flex justify-between text-xs">
              <span className="text-zinc-600">
                Peak bandwidth
              </span>

              <span className="text-zinc-300">
                {Number(data.peak_bandwidth).toLocaleString()} Kbps
              </span>
            </div>

          </div>

          <div className="mt-8 border-t border-white/[0.06] pt-5">

            <div className="flex items-center gap-2">

              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" />

                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
              </span>

              <span className="text-xs text-zinc-500">
                SDN protection layer operational
              </span>

            </div>

          </div>

        </div>

      </div>

    </div>
  );
}

export default Dashboard;