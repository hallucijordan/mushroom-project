import type { QueryStrategy } from '../../types/training';
import { getStrategyLabel } from '../../utils/formatters';

interface Props {
  value: QueryStrategy;
  onChange: (s: QueryStrategy) => void;
  disabled?: boolean;
  lockedUntilLevel?: number;
  currentLevel?: number;
}

const STRATEGIES: { value: QueryStrategy; icon: string; desc: string; minLevel: number }[] = [
  { value: 'random',      icon: '🎲', desc: 'Pick randomly from the pool',                       minLevel: 0 },
  { value: 'uncertainty', icon: '❓', desc: 'Focus on samples the model is least sure about',    minLevel: 1 },
  { value: 'committee',   icon: '🗳️', desc: 'Pick where a committee of models disagrees most',  minLevel: 1 },
  { value: 'bald',        icon: '🧠', desc: 'Bayesian Active Learning by Disagreement',          minLevel: 1 },
];

export function StrategyPicker({ value, onChange, disabled, currentLevel = 5 }: Props) {
  return (
    <div className="space-y-2">
      <p className="uppercase tracking-wide" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Query Strategy</p>
      {STRATEGIES.map((s) => {
        const locked = currentLevel < s.minLevel;
        const selected = value === s.value;
        return (
          <button
            key={s.value}
            disabled={disabled || locked}
            onClick={() => onChange(s.value)}
            className="w-full text-left px-3 py-2.5 rounded-lg border transition-all flex items-center gap-3 font-body"
            style={{
              background: selected ? 'rgba(226,163,70,0.15)' : locked ? 'var(--bg-deep)' : 'var(--bg-raised)',
              borderColor: selected ? 'var(--accent)' : 'var(--border)',
              color: selected ? 'var(--accent)' : locked ? 'var(--border)' : 'var(--text)',
              cursor: locked ? 'not-allowed' : 'pointer',
              opacity: locked ? 0.4 : 1,
            }}
          >
            <span className="text-lg">{s.icon}</span>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium leading-none mb-0.5">
                {getStrategyLabel(s.value)}
                {locked && <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)' }}>🔒 Lv.{s.minLevel}+</span>}
              </p>
              <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>{s.desc}</p>
            </div>
            {selected && <span className="shrink-0" style={{ color: 'var(--accent)' }}>&#x2714;</span>}
          </button>
        );
      })}
    </div>
  );
}
