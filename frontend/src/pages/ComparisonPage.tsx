import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { api } from '../services/api';
import { useCatStore } from '../stores/catStore';
import { CatAvatar } from '../components/cat/CatAvatar';
import { CatDialogue } from '../components/cat/CatDialogue';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import type { ComparisonRun } from '../types/mushroom';
import { formatAccuracy } from '../utils/formatters';

const STRATEGIES = [
  { value: 'random',      label: 'Random',      icon: '🎲', color: '#94a3b8' },
  { value: 'uncertainty', label: 'Uncertainty',  icon: '❓', color: '#e2a346' },
  { value: 'committee',   label: 'QBC',          icon: '🗳️', color: '#60a5fa' },
  { value: 'bald',        label: 'BALD',         icon: '🧠', color: '#a78bfa' },
];

const LEARNERS = [
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'logistic',      label: 'Logistic Regression' },
  { value: 'mlp',           label: 'Neural Network' },
];

export function ComparisonPage() {
  const { level, name } = useCatStore();
  const [selected, setSelected] = useState<string[]>(['random', 'uncertainty']);
  const [learner, setLearner] = useState('random_forest');
  const [budget, setBudget] = useState(2000);
  const [runs, setRuns] = useState<ComparisonRun[]>([]);
  const [loading, setLoading] = useState(false);

  const toggle = (s: string) => {
    setSelected((prev) =>
      prev.includes(s) ? (prev.length > 1 ? prev.filter((x) => x !== s) : prev) : [...prev, s]
    );
  };

  const run = async () => {
    setLoading(true);
    const results = await api.runComparison(selected, learner, budget);
    setRuns(results);
    setLoading(false);
  };

  const chartData = runs.length > 0
    ? runs[0].curve.map((pt, i) => {
        const point: Record<string, number> = { labels: pt.labels };
        runs.forEach((r) => { point[r.strategy] = Math.round((r.curve[i]?.accuracy ?? 0) * 100); });
        return point;
      })
    : [];

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-lg mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)' }}>Strategy Showdown</h1>
        <p className="font-body" style={{ color: 'var(--text-muted)' }}>Compare how different active learning strategies train {name}.</p>
        {level < 4 && (
          <div className="mt-3 inline-flex items-center gap-2 border px-4 py-2 rounded-lg text-sm font-body" style={{ background: 'rgba(226,163,70,0.1)', borderColor: 'var(--accent)', color: 'var(--accent)' }}>
            🔒 Fully unlocks at Level 4 (Myco Apprentice) — previewing now
          </div>
        )}
      </div>

      {/* Cat */}
      <div className="flex items-center gap-4 rounded-xl border p-5 mb-6" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <CatAvatar mood={loading ? 'thinking' : runs.length > 0 ? 'proud' : 'idle'} level={level} size={80} />
        <CatDialogue
          text={
            loading
              ? "Running simulations... this may take a moment!"
              : runs.length > 0
              ? `Fascinating! ${runs.find(r => r.strategy === 'uncertainty')?.labelsToReach95 != null
                  ? `Uncertainty sampling reached 95% with ${runs.find(r => r.strategy === 'uncertainty')!.labelsToReach95} labels!`
                  : 'The results are in — look at those curves!'}`
              : "Choose your strategies and let's see which one teaches me the fastest!"
          }
          speakerName={name}
        />
      </div>

      {/* Config */}
      <div className="rounded-xl border p-5 mb-6 space-y-5" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div>
          <p className="uppercase tracking-wide mb-3" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Strategies to compare</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {STRATEGIES.map((s) => {
              const on = selected.includes(s.value);
              return (
                <button
                  key={s.value}
                  onClick={() => toggle(s.value)}
                  className="flex items-center gap-2 px-3 py-2.5 rounded-lg border text-sm font-medium transition-all font-body"
                  style={{
                    background: on ? 'rgba(226,163,70,0.15)' : 'var(--bg-raised)',
                    borderColor: on ? 'var(--accent)' : 'var(--border)',
                    color: on ? 'var(--accent)' : 'var(--text-muted)',
                  }}
                >
                  <span>{s.icon}</span>
                  <span className="truncate">{s.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 font-body">
          <div>
            <label className="block uppercase tracking-wide mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Classifier</label>
            <select
              value={learner}
              onChange={(e) => setLearner(e.target.value)}
              className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none"
              style={{ background: 'var(--bg-raised)', borderColor: 'var(--border)', color: 'var(--text)' }}
            >
              {LEARNERS.map((l) => <option key={l.value} value={l.value}>{l.label}</option>)}
            </select>
          </div>
          <div>
            <label className="block uppercase tracking-wide mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>
              Budget: {budget.toLocaleString()}
            </label>
            <input
              type="range"
              min={500} max={5000} step={500}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="w-full mt-2 accent-amber-500"
            />
          </div>
        </div>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={run}
          disabled={loading || selected.length === 0}
          className="w-full font-bold py-3 rounded-xl shadow-lg transition-colors disabled:opacity-40"
          style={{ background: 'var(--accent)', color: 'var(--bg-deep)', fontFamily: 'var(--pixel-font)', fontSize: 10 }}
        >
          {loading ? 'Running simulations...' : 'Run Comparison'}
        </motion.button>
      </div>

      {loading && (
        <div className="flex justify-center py-10">
          <LoadingSpinner size="lg" label="Simulating AL runs..." />
        </div>
      )}

      <AnimatePresence>
        {runs.length > 0 && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
            <div className="rounded-xl border p-5" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
              <h2 className="uppercase tracking-wide mb-4" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 8 }}>Learning Curves</h2>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="labels" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <YAxis domain={[40, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Accuracy']} contentStyle={{ fontSize: 12, borderRadius: 8, background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                  <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-muted)' }} />
                  <ReferenceLine y={95} stroke="var(--accent)" strokeDasharray="4 4" label={{ value: '95%', position: 'right', fontSize: 10, fill: 'var(--accent)' }} />
                  {runs.map((r) => {
                    const info = STRATEGIES.find((s) => s.value === r.strategy);
                    return (
                      <Line key={r.strategy} type="monotone" dataKey={r.strategy} name={info?.label ?? r.strategy} stroke={info?.color ?? '#888'} strokeWidth={2} dot={{ r: 3 }} />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {runs.map((r) => {
                const info = STRATEGIES.find((s) => s.value === r.strategy);
                return (
                  <div key={r.strategy} className="rounded-xl border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
                    <div className="flex items-center gap-2 mb-2">
                      <span style={{ color: info?.color }} className="text-xl">{info?.icon}</span>
                      <p className="font-semibold font-body" style={{ color: 'var(--text)' }}>{info?.label}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm font-body">
                      <div>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Final Accuracy</p>
                        <p className="font-bold" style={{ color: 'var(--accent)' }}>{formatAccuracy(r.finalAccuracy)}</p>
                      </div>
                      <div>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Labels to 95%</p>
                        <p className="font-bold" style={{ color: 'var(--text)' }}>{r.labelsToReach95?.toLocaleString() ?? 'Not reached'}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
