import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from 'recharts';

interface TimelineData {
  timestamp: number;
  score: number;
  frame: number;
}

interface TimelineChartProps {
  data: TimelineData[];
  title?: string;
}

export default function TimelineChart({
  data,
  title = 'Fake Score Timeline',
}: TimelineChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="bg-gray-100 rounded-lg p-8 text-center">
          <p className="text-gray-500">No timeline data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) => `${value.toFixed(1)}s`}
              stroke="#9ca3af"
              fontSize={12}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              stroke="#9ca3af"
              fontSize={12}
            />
            <Tooltip
              formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Fake Score']}
              labelFormatter={(label) => `Time: ${label.toFixed(2)}s`}
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
              }}
            />
            <ReferenceLine
              y={0.5}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{ value: 'Threshold', position: 'right', fill: '#f59e0b', fontSize: 12 }}
            />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#ef4444"
              fill="url(#colorScore)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full" />
            <span>Fake probability</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-amber-500" style={{ borderStyle: 'dashed' }} />
            <span>Threshold (50%)</span>
          </div>
        </div>
        <span>{data.length} frames analyzed</span>
      </div>
    </div>
  );
}
