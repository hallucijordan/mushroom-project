import { useCatStore } from '../../stores/catStore';
import { CAT_LEVELS } from '../../types/cat';
import { LevelCard } from './LevelCard';
import { Achievements } from './Achievements';
import { CatProgressBadge } from '../cat/CatProgressBadge';
import { CatAvatar } from '../cat/CatAvatar';

export function JourneyMap() {
  const { level, mood, name, totalPredictions, correctPredictions } = useCatStore();

  return (
    <div className="max-w-2xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-lg mb-2" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)' }}>Training Journey</h1>
        <p className="font-body" style={{ color: 'var(--text-muted)' }}>{name}'s path from Clueless Kitten to Fungal Master</p>
      </div>

      {/* Cat summary */}
      <div className="flex items-center gap-5 rounded-xl border p-5 mb-6" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <CatAvatar mood={mood} level={level} size={120} />
        <div className="flex-1">
          <CatProgressBadge />
        </div>
      </div>

      {/* Stats */}
      {totalPredictions > 0 && (
        <div className="grid grid-cols-3 gap-3 mb-6 text-center">
          {[
            { label: 'Predictions', value: totalPredictions },
            { label: 'Correct', value: correctPredictions },
            { label: 'Accuracy', value: totalPredictions > 0 ? `${Math.round((correctPredictions / totalPredictions) * 100)}%` : '—' },
          ].map(({ label, value }) => (
            <div key={label} className="rounded-lg border p-3" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
              <p className="uppercase tracking-wide" style={{ color: 'var(--text-muted)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>{label}</p>
              <p className="font-bold text-lg font-body" style={{ color: 'var(--accent)' }}>{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Level cards */}
      <div className="space-y-3 mb-8">
        {CAT_LEVELS.map((info, i) => (
          <LevelCard key={info.level} info={info} currentLevel={level} index={i} />
        ))}
      </div>

      <Achievements />
    </div>
  );
}
