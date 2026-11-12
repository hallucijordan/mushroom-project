import { motion } from 'framer-motion';
import type { CatLevelInfo } from '../../types/cat';
import type { CatLevel } from '../../types/cat';
import { formatAccuracy } from '../../utils/formatters';

interface Props {
  info: CatLevelInfo;
  currentLevel: CatLevel;
  index: number;
}

export function LevelCard({ info, currentLevel, index }: Props) {
  const done = info.level < currentLevel;
  const active = info.level === currentLevel;
  const locked = info.level > currentLevel;

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.07 }}
      className="relative flex items-start gap-4 p-4 rounded-xl border transition-all"
      style={{
        background: active ? 'rgba(226,163,70,0.12)' : done ? 'rgba(74,222,128,0.08)' : 'var(--bg-card)',
        borderColor: active ? 'var(--accent)' : done ? 'rgba(74,222,128,0.3)' : 'var(--border)',
        opacity: locked ? 0.5 : 1,
      }}
    >
      {/* Level circle */}
      <div
        className="w-12 h-12 rounded-lg flex items-center justify-center text-lg font-bold shrink-0"
        style={{
          background: done ? 'var(--success)' : active ? 'var(--accent)' : 'var(--bg-raised)',
          color: done || active ? 'var(--bg-deep)' : 'var(--text-muted)',
          fontFamily: 'var(--pixel-font)',
          fontSize: 14,
        }}
      >
        {done ? '✓' : info.level}
      </div>

      <div className="flex-1 min-w-0 font-body">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="font-bold" style={{ color: 'var(--text)' }}>{info.title}</p>
          {active && (
            <span className="text-xs px-2 py-0.5 rounded-md" style={{ background: 'var(--accent)', color: 'var(--bg-deep)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Current</span>
          )}
        </div>
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
          Accuracy: {formatAccuracy(info.minAccuracy)}
          {info.level < 5 ? ` – ${formatAccuracy(info.maxAccuracy)}` : '+'}
        </p>
        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          Accessory: <span className="font-medium" style={{ color: 'var(--text)' }}>{info.accessory}</span>
        </p>
        <p className="text-xs mt-0.5" style={{ color: 'var(--accent)' }}>
          Unlocks: {info.unlocksFeature}
        </p>
      </div>

      {locked && (
        <span className="text-xl self-center" style={{ color: 'var(--text-muted)' }}>🔒</span>
      )}
    </motion.div>
  );
}
