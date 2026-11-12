import axios from 'axios';
import type { QueryStrategy, LearnerType, Prediction, LearningCurvePoint } from '../types/training';
import type { MushroomSample, ConsensusVerdict, ComparisonRun } from '../types/mushroom';
import type { CatMood } from '../types/cat';

const client = axios.create({ baseURL: '/api' });

export interface TrainStepResult {
  mushroom: MushroomSample;
  prediction: Prediction;
  trueLabel: 'edible' | 'poisonous';
  wasCorrect: boolean;
  newAccuracy: number;
  catReaction: CatMood;
  labeledCount: number;
  poolRemaining: number;
}

const CAP_SHAPES = ['convex', 'flat', 'knobbed', 'bell', 'sunken', 'conical'];
const CAP_COLORS = ['brown', 'buff', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'];
const HABITATS  = ['woods', 'grasses', 'leaves', 'paths', 'urban', 'waste', 'meadows'];
const SEASONS   = ['spring', 'summer', 'autumn', 'winter'];

function mockMushroom(): MushroomSample {
  return {
    index: Math.floor(Math.random() * 61000),
    features: {
      capShape:   CAP_SHAPES[Math.floor(Math.random() * CAP_SHAPES.length)],
      capColor:   CAP_COLORS[Math.floor(Math.random() * CAP_COLORS.length)],
      capSurface: ['smooth', 'scaly', 'fibrous', 'grooves'][Math.floor(Math.random() * 4)],
      gillColor:  ['buff', 'brown', 'gray', 'white', 'pink'][Math.floor(Math.random() * 5)],
      gillSpacing:['close', 'crowded', 'distant'][Math.floor(Math.random() * 3)],
      stemHeight: (4 + Math.random() * 12).toFixed(1),
      stemWidth:  (5 + Math.random() * 20).toFixed(0),
      habitat:    HABITATS[Math.floor(Math.random() * HABITATS.length)],
      season:     SEASONS[Math.floor(Math.random() * SEASONS.length)],
    },
  };
}

/** Model accuracy grows from 0.50 toward ~0.93 as labeledCount increases */
function mockAccuracy(labeled: number): number {
  const base = 0.50;
  const ceiling = 0.93;
  const scale = 0.003;
  return Math.min(ceiling, base + ceiling * (1 - Math.exp(-scale * labeled)));
}

/**
 * The UI is a prototype and is not fully connected to the backend yet.
 * In dev mode it uses simulated data unless VITE_USE_MOCK=false.
 * Simulated data is only used in mock mode: a failed API call is an error,
 * never a silent switch to fake results.
 */
export function isMockMode(): boolean {
  return import.meta.env.DEV && import.meta.env.VITE_USE_MOCK !== 'false';
}

async function safeCall<T>(fn: () => Promise<T>, mock: () => T): Promise<T> {
  if (isMockMode()) return mock();
  return fn();
}

export const api = {
  async startTraining(catName: string, strategy: QueryStrategy, learner: LearnerType) {
    return safeCall(
      async () => {
        const r = await client.post('/training/start', { cat_name: catName, strategy, learner });
        return { sessionId: r.data.session_id, labeledCount: r.data.labeled_count, poolRemaining: r.data.pool_remaining };
      },
      () => ({ sessionId: crypto.randomUUID(), labeledCount: 0, poolRemaining: 61000 })
    );
  },

  /**
   * One atomic training step:
   *   1. Pick next sample from pool (per strategy)
   *   2. Model predicts
   *   3. True label is revealed (known from dataset — no oracle needed)
   *   4. Model retrains
   * Returns everything the UI needs to animate the round.
   */
  async trainStep(sessionId: string, labeledCount: number): Promise<TrainStepResult> {
    return safeCall(
      async () => {
        const r = await client.post(`/training/${sessionId}/step`);
        return r.data as TrainStepResult;
      },
      () => {
        const mushroom = mockMushroom();
        const trueLabel: 'edible' | 'poisonous' = Math.random() > 0.48 ? 'edible' : 'poisonous';
        const accuracy = mockAccuracy(labeledCount);
        // Probability of a correct prediction matches current model accuracy
        const wasCorrect = Math.random() < accuracy;
        const predictedLabel: 'edible' | 'poisonous' = wasCorrect ? trueLabel : (trueLabel === 'edible' ? 'poisonous' : 'edible');
        const conf = wasCorrect ? 0.55 + Math.random() * 0.4 : 0.35 + Math.random() * 0.25;
        const newAccuracy = Math.max(0, Math.min(0.99, accuracy + (Math.random() - 0.48) * 0.01));

        const prediction: Prediction = {
          label: predictedLabel,
          confidence: conf,
          p_edible:     predictedLabel === 'edible' ? conf : 1 - conf,
          p_poisonous:  predictedLabel === 'poisonous' ? conf : 1 - conf,
          dialogue: wasCorrect
            ? conf > 0.8
              ? `Definitely ${predictedLabel}! The ${mushroom.features.capShape} cap gives it away.`
              : `I'll say ${predictedLabel}. The ${mushroom.features.habitat} habitat hints at it. (${Math.round(conf * 100)}%)`
            : `Hmm... I'll guess ${predictedLabel}? I'm not very sure. (${Math.round(conf * 100)}%)`,
        };

        return {
          mushroom,
          prediction,
          trueLabel,
          wasCorrect,
          newAccuracy,
          catReaction: wasCorrect ? 'happy' : (trueLabel === 'poisonous' ? 'sick' : 'confused'),
          labeledCount: labeledCount + 1,
          poolRemaining: Math.max(0, 61000 - labeledCount - 1),
        };
      }
    );
  },

  async getLearningCurve(sessionId: string): Promise<LearningCurvePoint[]> {
    return safeCall(
      async () => {
        const r = await client.get(`/training/${sessionId}/curve`);
        return r.data as LearningCurvePoint[];
      },
      () => []
    );
  },

  async setStrategy(sessionId: string, strategy: QueryStrategy) {
    return safeCall(
      async () => { await client.post(`/training/${sessionId}/strategy`, { strategy }); },
      () => {}
    );
  },

  async analyzePhoto(file: File): Promise<ConsensusVerdict> {
    const form = new FormData();
    form.append('file', file);
    return safeCall(
      async () => {
        const r = await client.post('/detect/analyze', form, { headers: { 'Content-Type': 'multipart/form-data' } });
        return r.data as ConsensusVerdict;
      },
      () => ({
        label: 'poisonous',
        confidence: 0.87,
        agreement: 'agree',
        catDialogue: "Both my analytical and visual sides agree: this looks POISONOUS. Don't eat it! The close gill spacing and brown convex cap are classic warning signs.",
        judges: [
          { judgeName: 'ML Model',      judgeType: 'ml_model' as const, label: 'poisonous' as const, confidence: 0.88, reasoning: 'Close gill spacing and brown convex cap match poisonous patterns' },
          { judgeName: 'Gemini Vision', judgeType: 'gemini' as const,   label: 'poisonous' as const, confidence: 0.86, reasoning: 'Visual texture and coloring suggest poisonous species' },
        ],
      } as ConsensusVerdict)
    );
  },

  async runComparison(strategies: string[], learner: string, budget: number): Promise<ComparisonRun[]> {
    return safeCall(
      async () => {
        const r = await client.post('/training/compare', { strategies, learner, budget });
        return r.data as ComparisonRun[];
      },
      () => {
        const SPEED: Record<string, number> = { uncertainty: 1.6, committee: 1.4, bald: 1.5, random: 1.0 };
        return strategies.map((strategy) => {
          const speed = SPEED[strategy] ?? 1.0;
          const steps = 10;
          const curve = Array.from({ length: steps }, (_, i) => {
            const labels = Math.round((budget / steps) * (i + 1));
            return { labels, accuracy: Math.min(0.97, 0.50 + (i / (steps - 1)) * 0.45 * speed + Math.random() * 0.02) };
          });
          const reached95 = curve.find((p) => p.accuracy >= 0.95);
          return {
            strategy, learner, budget,
            curve,
            finalAccuracy: curve[curve.length - 1].accuracy,
            labelsToReach95: reached95 ? reached95.labels : null,
          };
        });
      }
    );
  },
};
