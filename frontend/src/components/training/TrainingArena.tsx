import { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useCatStore } from '../../stores/catStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { api } from '../../services/api';
import type { TrainStepResult } from '../../services/api';
import { CatAvatar } from '../cat/CatAvatar';
import { CatDialogue } from '../cat/CatDialogue';
import { CatProgressBadge } from '../cat/CatProgressBadge';
import { LearningCurve } from './LearningCurve';
import { StrategyPicker } from './StrategyPicker';
import { MushroomCard } from '../shared/MushroomCard';
import { formatCount } from '../../utils/formatters';
import type { CatMood } from '../../types/cat';
import { CAT_LEVELS } from '../../types/cat';

type Speed = 'slow' | 'medium' | 'fast';

interface PhaseDurations { querying: number; predicting: number; reacting: number; retraining: number; }

const PHASE_DURATION: Record<Speed, PhaseDurations> = {
  fast:   { querying: 300,  predicting: 500,  reacting: 800,  retraining: 200 },
  medium: { querying: 600,  predicting: 1000, reacting: 1500, retraining: 400 },
  slow:   { querying: 1200, predicting: 2000, reacting: 2500, retraining: 600 },
};

function durations(speed: Speed): PhaseDurations {
  return PHASE_DURATION[speed];
}

interface RoundRecord {
  id: number;
  mushroom: TrainStepResult['mushroom'];
  predicted: 'edible' | 'poisonous';
  trueLabel: 'edible' | 'poisonous';
  wasCorrect: boolean;
  confidence: number;
}

export function TrainingArena() {
  const { level, name, setDialogue, setAccuracy, recordPrediction, addXP } = useCatStore();
  const {
    sessionId, phase, setPhase, strategy, setStrategy,
    labeledCount, poolRemaining, learningCurve,
    lastStep, applyStep, appendCurvePoint,
  } = useTrainingStore();

  const [catMood, setCatMood] = useState<CatMood>('idle');
  const [speed, setSpeed] = useState<Speed>('medium');
  const [history, setHistory] = useState<RoundRecord[]>([]);
  const roundIdRef = useRef(0);
  const runningRef = useRef(false);
  const stopRef    = useRef(false);

  const runLoop = useCallback(async () => {
    if (!sessionId || runningRef.current) return;
    runningRef.current = true;
    stopRef.current    = false;

    const sleep = (ms: number) => new Promise<void>((res) => setTimeout(res, ms));
    const currentLabeled = () => useTrainingStore.getState().labeledCount;

    while (!stopRef.current) {
      const d = durations(speed);

      setPhase('querying');
      setCatMood('curious');
      await sleep(d.querying);
      if (stopRef.current) break;

      setPhase('predicting');
      setCatMood('thinking');

      const step = await api.trainStep(sessionId, currentLabeled());
      setDialogue(step.prediction.dialogue);
      await sleep(d.predicting);
      if (stopRef.current) break;

      setPhase('reacting');
      applyStep(step);
      const mood = step.catReaction as CatMood;
      setCatMood(mood);
      recordPrediction(step.wasCorrect);
      addXP(step.wasCorrect ? 10 : 5);
      setAccuracy(step.newAccuracy);

      if (step.labeledCount % 10 === 0) {
        appendCurvePoint(step.labeledCount, step.newAccuracy);
      }

      roundIdRef.current += 1;
      const id = roundIdRef.current;
      setHistory((prev) => [
        { id, mushroom: step.mushroom, predicted: step.prediction.label,
          trueLabel: step.trueLabel, wasCorrect: step.wasCorrect, confidence: step.prediction.confidence },
        ...prev.slice(0, 5),
      ]);

      await sleep(d.reacting);
      if (stopRef.current) break;

      setPhase('retraining');
      setCatMood('thinking');
      await sleep(d.retraining);
    }

    setPhase('paused');
    setCatMood('idle');
    runningRef.current = false;
  }, [sessionId, speed, setPhase, setDialogue, applyStep, recordPrediction, addXP, setAccuracy, appendCurvePoint]);

  const handleStart = () => {
    stopRef.current = false;
    runLoop();
  };

  const handlePause = () => {
    stopRef.current = true;
  };

  useEffect(() => {
    const info = CAT_LEVELS[level];
    if (level > 0 && catMood === 'proud') {
      setDialogue(`I just reached ${info.title}! ${info.unlocksFeature} is now available!`);
    }
  }, [level, catMood, setDialogue]);

  const isRunning = phase !== 'idle' && phase !== 'paused';

  if (!sessionId) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <p className="text-5xl mb-4">🍄</p>
        <p className="text-xl font-bold mb-2" style={{ color: 'var(--text)', fontFamily: 'var(--pixel-font)', fontSize: 14 }}>No active training session</p>
        <p className="font-body mb-6" style={{ color: 'var(--text-muted)' }}>Start a session from the Home page to begin training {name}.</p>
        <a href="/" className="px-6 py-2.5 rounded-lg font-semibold transition-colors" style={{ background: 'var(--accent)', color: 'var(--bg-deep)', fontFamily: 'var(--pixel-font)', fontSize: 10 }}>
          Go to Home
        </a>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">

      {/* Stats bar */}
      <div className="grid grid-cols-4 gap-3 mb-6 text-center">
        {[
          { label: 'Labeled',  value: formatCount(labeledCount),  icon: '🏷️' },
          { label: 'Pool',     value: formatCount(poolRemaining),  icon: '🌊' },
          { label: 'Correct',  value: history.filter(r => r.wasCorrect).length + '/' + history.length, icon: '✅' },
          { label: 'Status',   value: phase === 'paused' || phase === 'idle' ? 'Paused' : phase, icon: '⚡' },
        ].map(({ label, value, icon }) => (
          <div key={label} className="rounded-lg border p-3" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
            <p className="text-lg">{icon}</p>
            <p className="uppercase tracking-wide" style={{ color: 'var(--text-muted)', fontFamily: 'var(--pixel-font)', fontSize: 6 }}>{label}</p>
            <p className="font-bold capitalize text-sm font-body" style={{ color: 'var(--text)' }}>{value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Left column: current mushroom + strategy */}
        <div className="flex flex-col gap-4">
          <div className="rounded-xl border min-h-[200px] flex items-center justify-center p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
            <AnimatePresence mode="wait">
              {lastStep?.mushroom ? (
                <motion.div
                  key={lastStep.mushroom.index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.3 }}
                >
                  <MushroomCard mushroom={lastStep.mushroom} />
                </motion.div>
              ) : (
                <div className="text-center">
                  <p className="text-5xl mb-2">🍄</p>
                  <p className="text-sm font-body" style={{ color: 'var(--text-muted)' }}>Waiting for first sample...</p>
                </div>
              )}
            </AnimatePresence>
          </div>

          {/* Last result badge */}
          <AnimatePresence>
            {lastStep && (
              <motion.div
                key={lastStep.labeledCount}
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-lg border px-4 py-3 text-sm flex items-center gap-3 font-body"
                style={{
                  background: lastStep.wasCorrect ? 'rgba(74,222,128,0.1)' : 'rgba(248,113,113,0.1)',
                  borderColor: lastStep.wasCorrect ? 'rgba(74,222,128,0.3)' : 'rgba(248,113,113,0.3)',
                  color: lastStep.wasCorrect ? 'var(--success)' : 'var(--danger)',
                }}
              >
                <span className="text-2xl">{lastStep.wasCorrect ? '😸' : lastStep.trueLabel === 'poisonous' ? '🤢' : '😕'}</span>
                <div>
                  <p className="font-semibold">{lastStep.wasCorrect ? 'Correct!' : 'Wrong!'}</p>
                  <p className="text-xs opacity-80">
                    Predicted <b>{lastStep.prediction.label}</b> &#x2192; actually <b>{lastStep.trueLabel}</b>
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <StrategyPicker value={strategy} onChange={setStrategy} disabled={isRunning} currentLevel={level} />
        </div>

        {/* Center: cat + controls */}
        <div className="flex flex-col items-center gap-4">
          {/* Cat — large sprite, no frame */}
          <div className="relative">
            <CatAvatar mood={catMood} level={level} size={240} />
            {isRunning && (
              <motion.div
                className="absolute inset-0 rounded-xl border-2 opacity-30 pointer-events-none"
                style={{ borderColor: 'var(--accent)' }}
                animate={{ scale: [1, 1.06, 1] }}
                transition={{ repeat: Infinity, duration: 1.2 }}
              />
            )}
          </div>

          <CatDialogue
            text={
              isRunning
                ? useCatStore.getState().currentDialogue
                : phase === 'paused'
                ? "Training paused. Click Start to resume!"
                : `Ready to learn! I'll study mushrooms using the ${strategy} strategy.`
            }
            speakerName={name}
          />

          {/* Phase label */}
          <div className="h-4" style={{ color: 'var(--text-muted)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>
            {phase === 'querying'   && 'Picking next specimen...'}
            {phase === 'predicting' && 'Predicting...'}
            {phase === 'reacting'   && (catMood === 'happy' ? 'Correct!' : catMood === 'sick' ? 'Wrong - poisonous!' : 'Wrong - was edible')}
            {phase === 'retraining' && 'Updating model...'}
          </div>

          {/* Speed control */}
          <div className="w-full max-w-xs">
            <p className="text-center mb-2 uppercase tracking-wide" style={{ color: 'var(--text-muted)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Speed</p>
            <div className="grid grid-cols-3 gap-1">
              {(['slow', 'medium', 'fast'] as Speed[]).map((s) => (
                <button
                  key={s}
                  disabled={isRunning}
                  onClick={() => setSpeed(s)}
                  className="py-1.5 rounded-lg text-xs font-medium capitalize transition-all font-body disabled:opacity-40"
                  style={{
                    background: speed === s ? 'var(--accent)' : 'var(--bg-raised)',
                    color: speed === s ? 'var(--bg-deep)' : 'var(--text-muted)',
                  }}
                >
                  {s === 'slow' ? 'Slow' : s === 'fast' ? 'Fast' : 'Med'}
                </button>
              ))}
            </div>
          </div>

          {/* Start / Pause */}
          <div className="w-full max-w-xs flex gap-3">
            {!isRunning ? (
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={handleStart}
                className="flex-1 font-bold py-3 rounded-xl shadow-lg transition-colors"
                style={{ background: 'var(--accent)', color: 'var(--bg-deep)', fontFamily: 'var(--pixel-font)', fontSize: 10 }}
              >
                {phase === 'paused' ? 'Resume' : 'Start Training'}
              </motion.button>
            ) : (
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={handlePause}
                className="flex-1 font-bold py-3 rounded-xl transition-colors"
                style={{ background: 'var(--bg-raised)', color: 'var(--text-muted)', fontFamily: 'var(--pixel-font)', fontSize: 10 }}
              >
                Pause
              </motion.button>
            )}
          </div>
        </div>

        {/* Right: progress + curve + history */}
        <div className="flex flex-col gap-4">
          <CatProgressBadge />

          {level >= 2 && <LearningCurve data={learningCurve} />}

          {/* Round history feed */}
          {history.length > 0 && (
            <div className="rounded-xl border p-3" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
              <p className="uppercase tracking-wide mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Recent Rounds</p>
              <div className="space-y-1">
                <AnimatePresence initial={false}>
                  {history.map((r) => (
                    <motion.div
                      key={r.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0 }}
                      className="flex items-center gap-2 text-xs px-2 py-1.5 rounded-lg font-body"
                      style={{
                        background: r.wasCorrect ? 'rgba(74,222,128,0.08)' : 'rgba(248,113,113,0.08)',
                        color: r.wasCorrect ? 'var(--success)' : 'var(--danger)',
                      }}
                    >
                      <span>{r.wasCorrect ? '✅' : r.trueLabel === 'poisonous' ? '☠️' : '❌'}</span>
                      <span className="capitalize">{r.mushroom.features.capShape} {r.mushroom.features.capColor}</span>
                      <span className="ml-auto opacity-60">
                        {r.predicted !== r.trueLabel && <span className="line-through mr-1">{r.predicted}</span>}
                        {r.trueLabel}
                      </span>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          )}

          {phase === 'idle' && history.length === 0 && (
            <div className="rounded-xl border p-4 text-sm font-body" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
              <p className="font-semibold mb-1" style={{ color: 'var(--accent)' }}>How it works</p>
              <p className="text-xs leading-relaxed">
                The warrior automatically studies mushrooms from the dataset. Since true labels are known, training runs
                without needing you to label anything — just watch Maodie learn!
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
