import { motion } from 'framer-motion';
import type { JudgeVerdict } from '../../types/mushroom';
import { formatConfidence } from '../../utils/formatters';

interface Props {
  verdict: JudgeVerdict;
  delay?: number;
}

export function JudgePanel({ verdict, delay = 0 }: Props) {
  const isPoison = verdict.label === 'poisonous';

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="rounded-xl border p-4 flex-1"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl">{verdict.judgeType === 'gemini' ? '🤖' : '🌲'}</span>
        <div>
          <p className="font-semibold text-sm font-body" style={{ color: 'var(--text)' }}>{verdict.judgeName}</p>
          <p className="text-xs capitalize font-body" style={{ color: 'var(--text-muted)' }}>{verdict.judgeType.replace('_', ' ')}</p>
        </div>
      </div>

      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-sm font-bold mb-3 font-body" style={{
        background: isPoison ? 'rgba(248,113,113,0.15)' : 'rgba(74,222,128,0.15)',
        color: isPoison ? 'var(--danger)' : 'var(--success)',
      }}>
        {isPoison ? '☠️' : '✅'} {verdict.label} — {formatConfidence(verdict.confidence)}
      </div>

      <div className="h-1.5 rounded-full overflow-hidden mb-3" style={{ background: 'var(--bg-raised)' }}>
        <motion.div
          className="h-full rounded-full"
          style={{ background: isPoison ? 'var(--danger)' : 'var(--success)' }}
          initial={{ width: 0 }}
          animate={{ width: `${verdict.confidence * 100}%` }}
          transition={{ duration: 0.8, delay: delay + 0.2 }}
        />
      </div>

      <p className="text-xs italic font-body" style={{ color: 'var(--text-muted)' }}>"{verdict.reasoning}"</p>
    </motion.div>
  );
}
