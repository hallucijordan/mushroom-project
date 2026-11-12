export function formatAccuracy(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatCount(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

export function getLevelProgressPercent(xp: number, level: number): number {
  const threshold = (level + 1) * 100;
  const base = level * 100;
  return Math.min(100, ((xp - base) / (threshold - base)) * 100);
}

export function getStrategyLabel(strategy: string): string {
  const labels: Record<string, string> = {
    random:      'Random',
    uncertainty: 'Uncertainty Sampling',
    committee:   'Query-by-Committee',
    bald:        'BALD',
    d_optimal:   'D-Optimal',
  };
  return labels[strategy] ?? strategy;
}

export function getLearnerLabel(learner: string): string {
  const labels: Record<string, string> = {
    random_forest: 'Random Forest',
    logistic:      'Logistic Regression',
    mlp:           'Neural Network',
  };
  return labels[learner] ?? learner;
}

export function getMushroomIcon(capShape?: string): string {
  // Simple deterministic emoji selection based on features
  const shapes: Record<string, string> = {
    convex: '🍄', flat: '🍁', knobbed: '🌰', bell: '🔔',
    sunken: '🫙', conical: '🔺',
  };
  return shapes[capShape ?? ''] ?? '🍄';
}
