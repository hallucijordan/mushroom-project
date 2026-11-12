import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../../services/api';
import { useCatStore } from '../../stores/catStore';
import { CatAvatar } from '../cat/CatAvatar';
import { CatDialogue } from '../cat/CatDialogue';
import { MushroomUploader } from '../training/MushroomUploader';
import { JudgePanel } from './JudgePanel';
import { ConsensusBar } from './ConsensusBar';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import type { ConsensusVerdict } from '../../types/mushroom';

export function ExpertMode() {
  const { level, name } = useCatStore();
  const [verdict, setVerdict] = useState<ConsensusVerdict | null>(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setVerdict(null);
    const result = await api.analyzePhoto(file);
    setVerdict(result);
    setLoading(false);
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-lg mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)' }}>Expert Mode</h1>
        <p className="font-body" style={{ color: 'var(--text-muted)' }}>Upload a mushroom photo — both the ML model and Gemini Vision will analyze it.</p>
        {level < 3 && (
          <div className="mt-3 inline-flex items-center gap-2 border px-4 py-2 rounded-lg text-sm font-body" style={{ background: 'rgba(226,163,70,0.1)', borderColor: 'var(--accent)', color: 'var(--accent)' }}>
            🔒 Fully unlocks at Level 3 (Gill Inspector) — previewing now
          </div>
        )}
      </div>

      {/* Cat */}
      <div className="flex flex-col items-center gap-4 mb-8">
        <CatAvatar mood={loading ? 'thinking' : verdict ? 'proud' : 'idle'} level={level} size={200} />
        <CatDialogue
          text={
            verdict
              ? verdict.catDialogue
              : loading
              ? "Hmm, let me consult both my analytical and visual sides..."
              : "Upload a mushroom photo and I'll give you my expert analysis!"
          }
          speakerName={name}
        />
      </div>

      {/* Upload */}
      <div className="rounded-xl border p-5 mb-5" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <MushroomUploader onFile={(f) => { setFile(f); setVerdict(null); }} />
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={analyze}
          disabled={!file || loading}
          className="mt-4 w-full font-semibold py-3 rounded-xl transition-colors shadow-lg disabled:opacity-40"
          style={{ background: 'var(--accent)', color: 'var(--bg-deep)', fontFamily: 'var(--pixel-font)', fontSize: 10 }}
        >
          {loading ? 'Analyzing...' : 'Analyze Mushroom'}
        </motion.button>
      </div>

      {loading && (
        <div className="flex justify-center py-8">
          <LoadingSpinner size="lg" label="Both judges are examining the specimen..." />
        </div>
      )}

      <AnimatePresence>
        {verdict && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="flex gap-4 mb-2">
              {verdict.judges.map((j, i) => (
                <JudgePanel key={j.judgeName} verdict={j} delay={i * 0.15} />
              ))}
            </div>
            <ConsensusBar verdict={verdict} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
