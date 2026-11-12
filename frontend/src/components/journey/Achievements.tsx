import { motion } from 'framer-motion';
import { ACHIEVEMENTS } from '../../types/training';
import { useSessionStore } from '../../stores/sessionStore';

export function Achievements() {
  const { unlockedAchievements } = useSessionStore();

  return (
    <div>
      <h2 className="text-sm mb-4" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)' }}>Achievements</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {ACHIEVEMENTS.map((a, i) => {
          const unlocked = unlockedAchievements.includes(a.key);
          return (
            <motion.div
              key={a.key}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.05 }}
              className="rounded-xl border p-3 text-center transition-all"
              style={{
                background: unlocked ? 'rgba(226,163,70,0.1)' : 'var(--bg-card)',
                borderColor: unlocked ? 'var(--accent)' : 'var(--border)',
                opacity: unlocked ? 1 : 0.4,
                filter: unlocked ? 'none' : 'grayscale(1)',
              }}
            >
              <p className="text-3xl mb-1">{a.icon}</p>
              <p className="text-xs font-semibold font-body" style={{ color: unlocked ? 'var(--accent)' : 'var(--text-muted)' }}>
                {a.label}
              </p>
              <p className="text-xs mt-0.5 leading-tight font-body" style={{ color: 'var(--text-muted)' }}>{a.description}</p>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
