export const MOOD_DURATION_MS = {
  idle:        0,
  thinking:    3000,
  happy:       1500,
  sick:        2000,
  proud:       2000,
  curious:     1500,
  sleeping:    0,
  confused:    1500,
  celebrating: 3000,
} as const;

export const TRANSITION = {
  fast:   { duration: 0.2 },
  normal: { duration: 0.4 },
  slow:   { duration: 0.8 },
  spring: { type: 'spring' as const, stiffness: 300, damping: 24 },
};
