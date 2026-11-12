import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { CatMood, CatLevel } from '../types/cat';
import { CAT_LEVELS } from '../types/cat';

interface CatStore {
  level: CatLevel;
  xp: number;
  mood: CatMood;
  name: string;
  accuracy: number;
  totalPredictions: number;
  correctPredictions: number;
  currentDialogue: string;

  setMood: (mood: CatMood) => void;
  addXP: (amount: number) => void;
  levelUp: () => void;
  setDialogue: (text: string) => void;
  setAccuracy: (accuracy: number) => void;
  setName: (name: string) => void;
  recordPrediction: (correct: boolean) => void;
  reset: () => void;
}

const INITIAL_STATE = {
  level: 0 as CatLevel,
  xp: 0,
  mood: 'idle' as CatMood,
  name: 'Professor Whiskers',
  accuracy: 0,
  totalPredictions: 0,
  correctPredictions: 0,
  currentDialogue: "Mew! I don't know anything about mushrooms yet, but I'm VERY hungry. Teach me?",
};

export const useCatStore = create<CatStore>()(
  persist(
    (set, get) => ({
      ...INITIAL_STATE,

      setMood: (mood) => set({ mood }),

      addXP: (amount) => {
        const { xp, level } = get();
        const newXP = xp + amount;
        const nextLevelThreshold = (level + 1) * 100;
        if (level < 5 && newXP >= nextLevelThreshold) {
          set({ xp: newXP, level: Math.min(5, level + 1) as CatLevel, mood: 'proud' });
        } else {
          set({ xp: newXP });
        }
      },

      levelUp: () => {
        const { level } = get();
        if (level < 5) {
          set({ level: (level + 1) as CatLevel, mood: 'proud' });
        }
      },

      setDialogue: (text) => set({ currentDialogue: text }),

      setAccuracy: (accuracy) => {
        const { level } = get();
        const info = CAT_LEVELS.find(
          (l) => accuracy >= l.minAccuracy && accuracy <= l.maxAccuracy
        );
        if (info && info.level !== level) {
          set({ accuracy, level: info.level, mood: 'proud' });
        } else {
          set({ accuracy });
        }
      },

      setName: (name) => set({ name }),

      recordPrediction: (correct) => {
        const { totalPredictions, correctPredictions } = get();
        set({
          totalPredictions: totalPredictions + 1,
          correctPredictions: correctPredictions + (correct ? 1 : 0),
        });
      },

      reset: () => set(INITIAL_STATE),
    }),
    { name: 'cat-store' }
  )
);
