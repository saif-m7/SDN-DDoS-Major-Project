import {
  LayoutDashboard,
  Activity,
  ShieldAlert,
  Crosshair,
  BarChart3,
  BrainCircuit,
  ShieldCheck,
} from "lucide-react";

const navigation = [
  {
    label: "Overview",
    icon: LayoutDashboard,
    path: "/",
  },
  {
    label: "Live Traffic",
    icon: Activity,
    path: "/traffic",
  },
  {
    label: "Detection",
    icon: ShieldAlert,
    path: "/detection",
  },
  {
    label: "Attacks",
    icon: Crosshair,
    path: "/attacks",
  },
  {
    label: "Analytics",
    icon: BarChart3,
    path: "/analytics",
  },
  {
    label: "Mitigation",
    icon: ShieldCheck,
    path: "/mitigation",
  },
];

function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-[250px] flex-col border-r border-white/[0.06] bg-[#0a0c10]">
      
      {/* Brand */}
      <div className="flex h-[76px] items-center border-b border-white/[0.06] px-6">
        <div>
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/10">
              <ShieldCheck className="h-5 w-5 text-cyan-400" />
            </div>

            <div>
              <h1 className="text-sm font-semibold tracking-wide text-white">
                SENTINEL
              </h1>

              <p className="text-[10px] uppercase tracking-[0.2em] text-zinc-600">
                SDN Security
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6">
        <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-600">
          Monitoring
        </p>

        <div className="space-y-1">
          {navigation.map((item) => {
            const Icon = item.icon;

            return (
              <a
                key={item.label}
                href={item.path}
                className="group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-zinc-500 transition-all duration-200 hover:bg-white/[0.04] hover:text-zinc-200"
              >
                <Icon className="h-[18px] w-[18px] transition-colors group-hover:text-cyan-400" />

                <span>{item.label}</span>
              </a>
            );
          })}
        </div>
      </nav>

      {/* Controller status */}
      <div className="border-t border-white/[0.06] p-4">
        <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
            </span>

            <span className="text-xs font-medium text-zinc-300">
              Controller Online
            </span>
          </div>

          <p className="mt-1 pl-4 text-[10px] text-zinc-600">
            Ryu SDN Controller
          </p>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;