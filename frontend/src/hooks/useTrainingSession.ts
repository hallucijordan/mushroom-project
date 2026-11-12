import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCatStore } from '../stores/catStore';
import { useTrainingStore } from '../stores/trainingStore';
import { useSessionStore } from '../stores/sessionStore';
import { api } from '../services/api';
import type { QueryStrategy, LearnerType } from '../types/training';

export function useTrainingSession() {
  const navigate = useNavigate();
  const { name } = useCatStore();
  const { setSessionId: storeSetSessionId, setStrategy, setLearner } = useTrainingStore();
  const { setSessionId, unlockAchievement } = useSessionStore();

  /**
   * Create a new training session and navigate to the training page.
   * The actual auto-training loop runs inside TrainingArena.
   */
  const begin = useCallback(async (strategy: QueryStrategy, learner: LearnerType) => {
    setStrategy(strategy);
    setLearner(learner);

    const { sessionId } = await api.startTraining(name, strategy, learner);
    storeSetSessionId(sessionId);
    setSessionId(sessionId);
    unlockAchievement('first_steps');

    navigate('/training');
  }, [name, setStrategy, setLearner, storeSetSessionId, setSessionId, unlockAchievement, navigate]);

  return { begin };
}
