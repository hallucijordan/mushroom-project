export type QueryStrategy = 'random' | 'uncertainty' | 'committee' | 'bald';
export type LearnerType = 'random_forest' | 'logistic' | 'mlp';
export type InitStrategy = 'random' | 'd_optimal';

export type TrainingPhase =
  | 'idle'
  | 'predicting'
  | 'waiting_reveal'
  | 'retraining'
  | 'querying';

export interface Prediction {
  label: 'edible' | 'poisonous';
  confidence: number;
  p_edible: number;
  p_poisonous: number;
  dialogue: string;
}

export interface LearningCurvePoint {
  labels: number;
  accuracy: number;
  f1?: number;
}

export interface TrainingSnapshot {
  labeledCount: number;
  cvAccuracy: number;
  cvStd: number;
  f1Score: number;
  rocAuc: number;
}

export interface AchievementInfo {
  key: string;
  label: string;
  description: string;
  icon: string;
  unlockedAt?: string;
}

export const ACHIEVEMENTS: AchievementInfo[] = [
  { key: 'first_steps',      label: 'First Steps',       description: 'Label your first sample',               icon: '🐾' },
  { key: 'getting_warmer',   label: 'Getting Warmer',     description: 'Reach 60% accuracy',                   icon: '🌡️' },
  { key: 'spore_sense',      label: 'Spore Sense',        description: 'Reach 75% accuracy',                   icon: '🔬' },
  { key: 'lab_certified',    label: 'Lab Certified',      description: 'Reach 85% accuracy',                   icon: '🥼' },
  { key: 'sharp_eyes',       label: 'Sharp Eyes',         description: '10 correct predictions in a row',      icon: '👁️' },
  { key: 'poison_survivor',  label: 'Poison Survivor',    description: 'Survive 5 false negatives',            icon: '💚' },
  { key: 'speed_learner',    label: 'Speed Learner',      description: 'Reach Level 3 under 500 labels',       icon: '⚡' },
  { key: 'strategist',       label: 'Strategist',         description: 'Try all 4 query strategies',           icon: '🧠' },
  { key: 'photo_detective',  label: 'Photo Detective',    description: 'Upload 10 real mushroom photos',       icon: '📸' },
  { key: 'fungal_phd',       label: 'Fungal PhD',         description: 'Reach Level 5',                        icon: '🎓' },
  { key: 'the_full_course',  label: 'The Full Course',    description: 'Complete strategy comparison',          icon: '📊' },
];
