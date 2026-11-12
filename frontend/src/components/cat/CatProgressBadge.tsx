import { motion } from 'framer-motion';
import { useCatStore } from '../../stores/catStore';
import { CAT_LEVELS } from '../../types/cat';
import { formatAccuracy, getLevelProgressPercent } from '../../utils/formatters';

export function CatProgressBadge() {
  const { level, xp, accuracy, totalPredictions, correctPredictions } = useCatStore();
  const info = CAT_LEVELS[level];
  const progress = getLevelProgressPercent(xp, level);
  const nextInfo = CAT_LEVELS[Math.min(5, level + 1)];

  return (
    <div className="rounded-xl border p-4 w-full" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <div className="flex items-center justify-between mb-2">
        <div>
          <p className="uppercase tracking-wide" style={{ color: 'var(--text-muted)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Level {level}</p>
          <p className="font-bold font-body" style={{ color: 'var(--accent)' }}>{info.title}</p>
        </div>
        <div className="text-right">
          <p className="text-xs font-body" style={{ color: 'var(--text-muted)' }}>Accuracy</p>
          <p className="text-lg font-bold" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 14 }}>{formatAccuracy(accuracy)}</p>
        </div>
      </div>

      {/* XP bar */}
      <div className="mb-1">
        <div className="flex justify-between text-xs font-body mb-1" style={{ color: 'var(--text-muted)' }}>
          <span>{xp} XP</span>
          {level < 5 && <span>&#x2192; {nextInfo.title}</span>}
        </div>
        <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-raised)' }}>
          <motion.div
            className="h-full rounded-full"
            style={{ background: 'linear-gradient(90deg, var(--accent-dim), var(--accent))' }}
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </div>
      </div>

      {totalPredictions > 0 && (
        <p className="text-xs font-body mt-2 text-center" style={{ color: 'var(--text-muted)' }}>
          {correctPredictions}/{totalPredictions} correct predictions
        </p>
      )}
    </div>
  );
}
