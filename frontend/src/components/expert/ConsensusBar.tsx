import { motion } from 'framer-motion';
import type { ConsensusVerdict } from '../../types/mushroom';
import { formatConfidence } from '../../utils/formatters';

interface Props { verdict: ConsensusVerdict; }

export function ConsensusBar({ verdict }: Props) {
  const isPoison = verdict.label === 'poisonous';

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.4 }}
      className="rounded-xl border-2 p-5 mt-4"
      style={{
        background: isPoison ? 'rgba(248,113,113,0.08)' : 'rgba(74,222,128,0.08)',
        borderColor: isPoison ? 'rgba(248,113,113,0.4)' : 'rgba(74,222,128,0.4)',
      }}
    >
      <div className="flex items-center gap-3 mb-3">
        <span className="text-3xl">{isPoison ? '☠️' : '✅'}</span>
        <div>
          <p className="uppercase tracking-wide" style={{ color: 'var(--text-muted)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Final Verdict</p>
          <p className="text-2xl font-extrabold capitalize font-body" style={{ color: isPoison ? 'var(--danger)' : 'var(--success)' }}>
            {verdict.label}
          </p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-xs font-body" style={{ color: 'var(--text-muted)' }}>Confidence</p>
          <p className="text-xl font-bold" style={{ color: isPoison ? 'var(--danger)' : 'var(--success)', fontFamily: 'var(--pixel-font)', fontSize: 16 }}>
            {formatConfidence(verdict.confidence)}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 text-sm mb-3 font-body" style={{ color: verdict.agreement === 'agree' ? 'var(--text-muted)' : 'var(--accent)' }}>
        <span>{verdict.agreement === 'agree' ? '🤝' : '⚠️'}</span>
        <span>{verdict.agreement === 'agree' ? 'Both judges agree' : 'Judges disagree — use caution'}</span>
      </div>

      <div className="rounded-lg p-3 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <p className="text-xs font-medium mb-1" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Maodie says:</p>
        <p className="text-sm italic font-body" style={{ color: 'var(--text)' }}>"{verdict.catDialogue}"</p>
      </div>
    </motion.div>
  );
}
