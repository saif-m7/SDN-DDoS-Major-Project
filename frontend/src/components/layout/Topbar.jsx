import { Bell, Search, Radio } from "lucide-react";

function Topbar() {
  return (
    <header className="sticky top-0 z-30 flex h-[76px] items-center justify-between border-b border-white/[0.06] bg-[#07090d]/90 px-8 backdrop-blur-xl">
      
      {/* Left */}
      <div>
        <p className="text-xs text-zinc-600">
          Security Operations
        </p>

        <h2 className="mt-0.5 text-sm font-medium text-zinc-300">
          Network Security Console
        </h2>
      </div>

      {/* Right */}
      <div className="flex items-center gap-5">
        
        {/* Live indicator */}
        <div className="flex items-center gap-2 rounded-full border border-emerald-400/10 bg-emerald-400/[0.04] px-3 py-1.5">
          <Radio className="h-3.5 w-3.5 text-emerald-400" />

          <span className="text-[11px] font-medium text-emerald-400">
            LIVE
          </span>
        </div>

        {/* Search */}
        <button className="text-zinc-600 transition hover:text-zinc-300">
          <Search className="h-[18px] w-[18px]" />
        </button>

        {/* Notifications */}
        <button className="relative text-zinc-600 transition hover:text-zinc-300">
          <Bell className="h-[18px] w-[18px]" />

          <span className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-red-400" />
        </button>

        {/* User */}
        <div className="ml-2 flex h-8 w-8 items-center justify-center rounded-full border border-white/[0.08] bg-white/[0.04] text-xs font-medium text-zinc-400">
          S
        </div>
      </div>
    </header>
  );
}

export default Topbar;