import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SessionStore {
  sessionId: string | null;
  theme: 'light' | 'dark';
  soundEnabled: boolean;
  unlockedAchievements: string[];

  setSessionId: (id: string | null) => void;
  unlockAchievement: (key: string) => void;
  toggleSound: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
  reset: () => void;
}

export const useSessionStore = create<SessionStore>()(
  persist(
    (set, get) => ({
      sessionId: null,
      theme: 'light',
      soundEnabled: true,
      unlockedAchievements: [],

      setSessionId: (id) => set({ sessionId: id }),

      unlockAchievement: (key) => {
        const { unlockedAchievements } = get();
        if (!unlockedAchievements.includes(key)) {
          set({ unlockedAchievements: [...unlockedAchievements, key] });
        }
      },

      toggleSound: () => set((s) => ({ soundEnabled: !s.soundEnabled })),
      setTheme: (theme) => set({ theme }),
      reset: () => set({ sessionId: null, unlockedAchievements: [] }),
    }),
    { name: 'session-store' }
  )
);
