import type { MushroomSample } from '../../types/mushroom';
import { getMushroomIcon } from '../../utils/formatters';

interface Props {
  mushroom: MushroomSample;
  index?: number;
}

const FEATURE_LABELS: Record<string, string> = {
  capShape: 'Cap Shape', capColor: 'Cap Color', capSurface: 'Cap Surface',
  gillColor: 'Gill Color', gillSpacing: 'Gill Spacing',
  stemHeight: 'Stem Height (cm)', stemWidth: 'Stem Width (mm)',
  habitat: 'Habitat', season: 'Season',
};

export function MushroomCard({ mushroom, index }: Props) {
  const icon = getMushroomIcon(mushroom.features.capShape);
  const displayIndex = index ?? mushroom.index;

  return (
    <div className="rounded-xl border p-5 w-full max-w-xs" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-3xl">{icon}</span>
        <div>
          <p className="uppercase tracking-wide" style={{ color: 'var(--text-muted)', fontFamily: 'var(--pixel-font)', fontSize: 7 }}>Specimen</p>
          <p className="font-bold font-body" style={{ color: 'var(--accent)' }}>#{displayIndex.toLocaleString()}</p>
        </div>
      </div>

      <div className="space-y-1.5">
        {Object.entries(mushroom.features).map(([key, val]) => {
          if (!val) return null;
          return (
            <div key={key} className="flex justify-between text-sm font-body">
              <span style={{ color: 'var(--text-muted)' }}>{FEATURE_LABELS[key] ?? key}</span>
              <span className="font-medium capitalize" style={{ color: 'var(--text)' }}>{val}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
