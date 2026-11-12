import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { LearningCurvePoint } from '../../types/training';
import { formatAccuracy } from '../../utils/formatters';

interface Props {
  data: LearningCurvePoint[];
  targetAccuracy?: number;
}

export function LearningCurve({ data, targetAccuracy = 0.95 }: Props) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-sm rounded-xl border font-body" style={{ background: 'var(--bg-raised)', borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
        <p>Learning curve appears after first retrain cycle</p>
      </div>
    );
  }

  const formatted = data.map((d) => ({
    ...d,
    accuracyPct: Math.round(d.accuracy * 100),
    display: formatAccuracy(d.accuracy),
  }));

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-2">
        <p className="uppercase tracking-wide" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Learning Curve</p>
        {data.length > 0 && (
          <p className="text-sm font-semibold font-body" style={{ color: 'var(--accent)' }}>
            Current: {formatAccuracy(data[data.length - 1].accuracy)}
          </p>
        )}
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={formatted}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="labels"
            tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
            label={{ value: 'Labeled samples', position: 'insideBottomRight', offset: -5, fontSize: 10, fill: 'var(--text-muted)' }}
          />
          <YAxis
            domain={[40, 100]}
            tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            formatter={(v) => [`${v}%`, 'Accuracy']}
            labelFormatter={(l) => `${l} labels`}
            contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text)' }}
          />
          <ReferenceLine
            y={targetAccuracy * 100}
            stroke="var(--accent)"
            strokeDasharray="4 4"
            label={{ value: 'Target', position: 'right', fontSize: 10, fill: 'var(--accent)' }}
          />
          <Line
            type="monotone"
            dataKey="accuracyPct"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={{ r: 3, fill: 'var(--accent)' }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
