import { create } from 'zustand';
import type { QueryStrategy, LearnerType, LearningCurvePoint } from '../types/training';
import type { TrainStepResult } from '../services/api';

export type AutoPhase =
  | 'idle'       // not started
  | 'querying'   // picking next sample (curious)
  | 'predicting' // model predicts (thinking)
  | 'reacting'   // showing result (happy/sick)
  | 'retraining' // model retraining
  | 'paused';    // user paused

interface TrainingStore {
  sessionId: string | null;
  phase: AutoPhase;
  strategy: QueryStrategy;
  learner: LearnerType;
  labeledCount: number;
  poolRemaining: number;
  learningCurve: LearningCurvePoint[];
  lastStep: TrainStepResult | null;

  setSessionId: (id: string) => void;
  setStrategy: (s: QueryStrategy) => void;
  setLearner: (l: LearnerType) => void;
  setPhase: (phase: AutoPhase) => void;
  applyStep: (step: TrainStepResult) => void;
  appendCurvePoint: (labeled: number, accuracy: number) => void;
  reset: () => void;
}

const INITIAL = {
  sessionId: null as string | null,
  phase: 'idle' as AutoPhase,
  strategy: 'random' as QueryStrategy,
  learner: 'random_forest' as LearnerType,
  labeledCount: 0,
  poolRemaining: 61000,
  learningCurve: [] as LearningCurvePoint[],
  lastStep: null as TrainStepResult | null,
};

export const useTrainingStore = create<TrainingStore>((set) => ({
  ...INITIAL,

  setSessionId: (id) => set({ sessionId: id }),
  setStrategy:  (strategy) => set({ strategy }),
  setLearner:   (learner)  => set({ learner }),
  setPhase:     (phase)    => set({ phase }),

  applyStep: (step) =>
    set({
      lastStep: step,
      labeledCount: step.labeledCount,
      poolRemaining: step.poolRemaining,
    }),

  appendCurvePoint: (labels, accuracy) =>
    set((s) => ({
      learningCurve: [...s.learningCurve, { labels, accuracy }],
    })),

  reset: () => set(INITIAL),
}));
