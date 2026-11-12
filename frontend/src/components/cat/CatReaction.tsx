import { motion, AnimatePresence } from 'framer-motion';
import type { CatMood } from '../../types/cat';

interface Props {
  mood: CatMood;
  visible: boolean;
}

const REACTION_CONTENT: Partial<Record<CatMood, { label: string; color: string; particles: string[] }>> = {
  happy:       { label: 'Correct!',  color: 'text-green-600',  particles: ['⭐', '✨', '🌟'] },
  sick:        { label: 'Oops!',     color: 'text-green-700',  particles: ['💚', '🤢', '☠️'] },
  proud:       { label: 'Level Up!', color: 'text-yellow-600', particles: ['🎉', '🏆', '⭐'] },
  celebrating: { label: 'Expert!',   color: 'text-pink-600',   particles: ['🎊', '🎉', '🏅'] },
};

export function CatReaction({ mood, visible }: Props) {
  const content = REACTION_CONTENT[mood];
  if (!content) return null;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, scale: 0.5, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.5, y: -20 }}
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
        >
          <div className={`text-2xl font-bold ${content.color} flex gap-1`}>
            {content.particles.map((p, i) => (
              <motion.span
                key={i}
                initial={{ y: 0, opacity: 1 }}
                animate={{ y: -40, opacity: 0 }}
                transition={{ delay: i * 0.1, duration: 0.8 }}
              >
                {p}
              </motion.span>
            ))}
          </div>
          <motion.p
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className={`absolute text-xl font-extrabold ${content.color} drop-shadow-md`}
          >
            {content.label}
          </motion.p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
