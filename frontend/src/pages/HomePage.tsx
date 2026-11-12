import { useState } from 'react';
import { motion } from 'framer-motion';
import { useCatStore } from '../stores/catStore';
import { useTrainingStore } from '../stores/trainingStore';
import { useTrainingSession } from '../hooks/useTrainingSession';
import { CatDialogue } from '../components/cat/CatDialogue';
import type { QueryStrategy, LearnerType } from '../types/training';
import { getLearnerLabel } from '../utils/formatters';
import heroImg from '../assets/maodie/hero.png';

const LEARNERS: { value: LearnerType; icon: string; desc: string }[] = [
  { value: 'random_forest', icon: '🌲', desc: 'Reliable and fast — a committee of decision trees' },
  { value: 'logistic',      icon: '📈', desc: 'Simple and well-calibrated probabilities' },
  { value: 'mlp',           icon: '🧠', desc: 'Neural network — powerful but slower to train' },
];

const INIT_STRATEGIES: { value: 'random' | 'd_optimal'; label: string; desc: string }[] = [
  { value: 'random',      label: 'Random Start',   desc: 'Pick initial samples randomly (simple)' },
  { value: 'd_optimal',   label: 'D-Optimal Start', desc: 'Maximize initial coverage (smarter)' },
];

export function HomePage() {
  const { name, setName, reset: resetCat } = useCatStore();
  const { reset: resetTraining } = useTrainingStore();
  const { begin } = useTrainingSession();

  const [catName, setCatName] = useState(name);
  const [strategy, setStrategy] = useState<QueryStrategy>('random');
  const [learner, setLearner] = useState<LearnerType>('random_forest');
  const [loading, setLoading] = useState(false);

  const handleStart = async () => {
    setLoading(true);
    resetCat();
    resetTraining();
    if (catName.trim()) setName(catName.trim());
    await begin(strategy, learner);
    setLoading(false);
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      {/* Hero */}
      <div className="text-center mb-10">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="flex justify-center mb-4"
        >
          <img
            src={heroImg}
            alt="Maodie the Mushroom Warrior"
            className="w-72 h-72 object-contain drop-shadow-[0_0_40px_rgba(226,163,70,0.3)]"
            style={{ imageRendering: 'pixelated' }}
          />
        </motion.div>

        <CatDialogue
          text={`I am ${name}... I know nothing of mushrooms yet, but I have sworn a warrior's oath! Train me, master!`}
          speakerName={name}
        />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-8"
        >
          <h1 className="text-xl mb-3" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)' }}>
            MushRoom Academy
          </h1>
          <p className="font-body text-base max-w-md mx-auto" style={{ color: 'var(--text-muted)' }}>
            Train a mushroom warrior to identify poisonous fungi using <strong style={{ color: 'var(--accent)' }}>Active Learning</strong>.
            Every label you give is a real data point.
          </p>
        </motion.div>
      </div>

      {/* Setup card */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="rounded-2xl border p-6 space-y-6"
        style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
      >
        {/* Cat name */}
        <div>
          <label className="block uppercase tracking-wide mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 8 }}>
            Name your warrior
          </label>
          <input
            value={catName}
            onChange={(e) => setCatName(e.target.value)}
            className="w-full rounded-lg px-4 py-2.5 font-body border focus:outline-none transition"
            style={{ background: 'var(--bg-raised)', borderColor: 'var(--border)', color: 'var(--text)' }}
            placeholder="Maodie"
            maxLength={30}
          />
        </div>

        {/* Classifier */}
        <div>
          <label className="block uppercase tracking-wide mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 8 }}>
            Choose warrior's brain
          </label>
          <div className="space-y-2">
            {LEARNERS.map((l) => (
              <button
                key={l.value}
                onClick={() => setLearner(l.value)}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg border text-left transition-all font-body"
                style={{
                  background: learner === l.value ? 'rgba(226,163,70,0.15)' : 'var(--bg-raised)',
                  borderColor: learner === l.value ? 'var(--accent)' : 'var(--border)',
                  color: learner === l.value ? 'var(--accent)' : 'var(--text)',
                }}
              >
                <span className="text-xl">{l.icon}</span>
                <div>
                  <p className="font-medium text-sm">{getLearnerLabel(l.value)}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{l.desc}</p>
                </div>
                {learner === l.value && <span className="ml-auto" style={{ color: 'var(--accent)' }}>&#x2714;</span>}
              </button>
            ))}
          </div>
        </div>

        {/* Initial strategy */}
        <div>
          <label className="block uppercase tracking-wide mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 8 }}>
            Starting query strategy
          </label>
          <div className="grid grid-cols-2 gap-2">
            {INIT_STRATEGIES.map((s) => (
              <button
                key={s.value}
                onClick={() => setStrategy(s.value as QueryStrategy)}
                className="flex flex-col px-3 py-3 rounded-lg border text-left transition-all font-body"
                style={{
                  background: strategy === s.value ? 'rgba(226,163,70,0.15)' : 'var(--bg-raised)',
                  borderColor: strategy === s.value ? 'var(--accent)' : 'var(--border)',
                  color: strategy === s.value ? 'var(--accent)' : 'var(--text)',
                }}
              >
                <p className="font-medium text-sm">{s.label}</p>
                <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{s.desc}</p>
              </button>
            ))}
          </div>
        </div>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleStart}
          disabled={loading}
          className="w-full font-bold py-4 rounded-xl shadow-lg transition-colors disabled:opacity-40"
          style={{ background: 'var(--accent)', color: 'var(--bg-deep)', fontFamily: 'var(--pixel-font)', fontSize: 12 }}
        >
          {loading ? 'Setting up...' : 'Start Training!'}
        </motion.button>
      </motion.div>

      {/* Feature previews */}
      <div className="grid grid-cols-3 gap-3 mt-8">
        {[
          { icon: '🎮', title: 'Interactive Training', desc: 'Watch the warrior learn in real time' },
          { icon: '📊', title: 'Live Learning Curves', desc: 'See accuracy improve over rounds' },
          { icon: '🔬', title: 'Expert Analysis', desc: 'ML model + Gemini Vision judges' },
        ].map(({ icon, title, desc }) => (
          <div key={title} className="text-center rounded-xl border p-4" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
            <p className="text-3xl mb-2">{icon}</p>
            <p className="text-xs font-semibold" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>{title}</p>
            <p className="text-xs mt-1 font-body" style={{ color: 'var(--text-muted)' }}>{desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
