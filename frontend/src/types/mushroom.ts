export interface MushroomFeatures {
  capShape?: string;
  capColor?: string;
  capSurface?: string;
  gillColor?: string;
  gillSpacing?: string;
  stemHeight?: string;
  stemWidth?: string;
  habitat?: string;
  season?: string;
  [key: string]: string | undefined;
}

export interface MushroomSample {
  index: number;
  features: MushroomFeatures;
  displayIcon?: string; // SVG path derived from features
}

export interface JudgeVerdict {
  judgeName: string;
  judgeType: 'ml_model' | 'gemini';
  label: 'edible' | 'poisonous';
  confidence: number;
  reasoning: string;
  features?: MushroomFeatures;
}

export interface ConsensusVerdict {
  label: 'edible' | 'poisonous';
  confidence: number;
  agreement: 'agree' | 'disagree';
  catDialogue: string;
  judges: JudgeVerdict[];
}

export interface ComparisonRun {
  strategy: string;
  learner: string;
  budget: number;
  curve: { labels: number; accuracy: number }[];
  finalAccuracy: number;
  labelsToReach95: number | null;
}
