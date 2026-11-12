import { useEffect, useRef } from 'react';
import { useCatStore } from '../stores/catStore';
import type { CatMood } from '../types/cat';
import { MOOD_DURATION_MS } from '../utils/animations';

export function useCatAnimation() {
  const { mood, setMood } = useCatStore();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerMood = (newMood: CatMood, returnTo: CatMood = 'idle') => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setMood(newMood);
    const duration = MOOD_DURATION_MS[newMood];
    if (duration > 0) {
      timerRef.current = setTimeout(() => setMood(returnTo), duration);
    }
  };

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  return { mood, triggerMood };
}
