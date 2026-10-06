import { ArrowUpRight } from "lucide-react";

function StatCard({ label, value, unit, icon: Icon, accent = "cyan" }) {
  const accents = {
    cyan: "text-cyan-400 bg-cyan-400/10 border-cyan-400/10",
    emerald: "text-emerald-400 bg-emerald-400/10 border-emerald-400/10",
    red: "text-red-400 bg-red-400/10 border-red-400/10",
    amber: "text-amber-400 bg-amber-400/10 border-amber-400/10",
  };

  return (
    <div className="group relative overflow-hidden rounded-xl border border-white/[0.06] bg-[#0b0e13] p-5 transition-all duration-300 hover:border-white/[0.12]">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-zinc-600">
            {label}
          </p>

          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-2xl font-semibold tracking-tight text-zinc-100">
              {value}
            </span>

            {unit && (
              <span className="text-xs text-zinc-600">
                {unit}
              </span>
            )}
          </div>
        </div>

        <div
          className={`flex h-9 w-9 items-center justify-center rounded-lg border ${accents[accent]}`}
        >
          <Icon className="h-4 w-4" />
        </div>
      </div>

      <div className="mt-5 flex items-center gap-1 text-[10px] text-zinc-600">
        <ArrowUpRight className="h-3 w-3" />
        <span>Live network data</span>
      </div>
    </div>
  );
}

export default StatCard;