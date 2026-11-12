import { motion, AnimatePresence } from 'framer-motion';
import type { CatMood, CatLevel } from '../../types/cat';
import { TRANSITION } from '../../utils/animations';

import idleImg from '../../assets/maodie/idle.png';
import thinkingImg from '../../assets/maodie/thinking.png';
import happyImg from '../../assets/maodie/happy.png';
import sickImg from '../../assets/maodie/sick.png';
import proudImg from '../../assets/maodie/proud.png';

interface Props {
  mood: CatMood;
  level: CatLevel;
  size?: number;
}

/** Only 5 real sprites — everything else falls back. */
const MOOD_SPRITE: Record<CatMood, string> = {
  idle:        idleImg,
  thinking:    thinkingImg,
  happy:       happyImg,
  sick:        sickImg,
  proud:       proudImg,
  curious:     thinkingImg,
  sleeping:    idleImg,
  confused:    thinkingImg,
  celebrating: happyImg,
};

const MOOD_ANIMATION: Record<CatMood, object> = {
  idle:        { y: [0, -4, 0], transition: { repeat: Infinity, duration: 3, ease: 'easeInOut' } },
  thinking:    { rotate: [-2, 2, -2], transition: { repeat: Infinity, duration: 0.8 } },
  happy:       { y: [0, -14, 0], transition: { repeat: 3, duration: 0.4 } },
  sick:        { x: [-5, 5, -5], transition: { repeat: 3, duration: 0.3 } },
  proud:       { scale: [1, 1.08, 1], transition: { repeat: 2, duration: 0.6 } },
  curious:     { rotate: [0, -6, 0], transition: { repeat: 2, duration: 0.5 } },
  sleeping:    { y: [0, -2, 0], transition: { repeat: Infinity, duration: 4, ease: 'easeInOut' } },
  confused:    { rotate: [-10, 0], transition: { duration: 0.4 } },
  celebrating: { y: [0, -18, 0], scale: [1, 1.06, 1], transition: { repeat: Infinity, duration: 0.6 } },
};

export function CatAvatar({ mood, level, size = 200 }: Props) {
  const sprite = MOOD_SPRITE[mood];

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <AnimatePresence mode="wait">
        <motion.img
          key={mood}
          src={sprite}
          alt={`Maodie — ${mood}`}
          className="w-full h-full object-contain drop-shadow-[0_4px_24px_rgba(255,180,50,0.35)]"
          style={{ imageRendering: 'pixelated' }}
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1, ...MOOD_ANIMATION[mood] }}
          exit={{ opacity: 0, scale: 0.85 }}
          transition={TRANSITION.spring}
        />
      </AnimatePresence>

      {/* Level badge — pixel style */}
      <div
        className="absolute -top-1 -right-1 bg-amber-400 text-[#1a1a2e] text-xs font-bold rounded-md w-7 h-7 flex items-center justify-center shadow-lg border-2 border-amber-300"
        style={{ fontFamily: "'Press Start 2P', monospace", fontSize: 10 }}
      >
        {level}
      </div>
    </div>
  );
}
