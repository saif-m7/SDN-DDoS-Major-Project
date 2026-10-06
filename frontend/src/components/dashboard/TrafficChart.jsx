import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

function TrafficChart({ data = [] }) {
  const chartData = data
    .slice()
    .reverse()
    .map((item, index) => ({
      index: index + 1,
      bandwidth: Number(item.total_kbps || 0),
    }));

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
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
            <linearGradient id="trafficFill" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="0%"
                stopColor="#22d3ee"
                stopOpacity={0.20}
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
            dataKey="index"
            tick={{ fill: "#52525b", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />

          <YAxis
            tick={{ fill: "#52525b", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />

          <Tooltip
            contentStyle={{
              backgroundColor: "#0b0e13",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: "8px",
              color: "#e4e4e7",
              fontSize: "12px",
            }}
            formatter={(value) => [
              `${Number(value).toFixed(2)} Kbps`,
              "Bandwidth",
            ]}
            labelFormatter={(label) => `Sample ${label}`}
          />

          <Area
            type="monotone"
            dataKey="bandwidth"
            stroke="#22d3ee"
            strokeWidth={2}
            fill="url(#trafficFill)"
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
  );
}

export default TrafficChart;