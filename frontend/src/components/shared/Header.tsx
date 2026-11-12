import { Link, useLocation } from 'react-router-dom';
import { useCatStore } from '../../stores/catStore';
import { CAT_LEVELS } from '../../types/cat';
import { formatAccuracy } from '../../utils/formatters';
import idleImg from '../../assets/maodie/idle.png';

const NAV = [
  { path: '/',           label: 'Home' },
  { path: '/training',   label: 'Train' },
  { path: '/expert',     label: 'Expert' },
  { path: '/journey',    label: 'Journey' },
  { path: '/comparison', label: 'Compare' },
];

export function Header() {
  const { pathname } = useLocation();
  const { level, accuracy, name } = useCatStore();
  const levelInfo = CAT_LEVELS[level];

  return (
    <header className="sticky top-0 z-50 border-b shadow-lg" style={{ background: 'var(--bg-surface)', borderColor: 'var(--border)' }}>
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center gap-6">
        <Link to="/" className="flex items-center gap-2 font-bold shrink-0" style={{ color: 'var(--accent)', fontFamily: 'var(--pixel-font)', fontSize: 11 }}>
          <img src={idleImg} alt="Maodie" className="w-8 h-8 object-contain" style={{ imageRendering: 'pixelated' }} />
          <span className="hidden sm:inline">MushRoom Academy</span>
        </Link>

        <nav className="flex gap-1 flex-1">
          {NAV.map(({ path, label }) => (
            <Link
              key={path}
              to={path}
              className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors font-body"
              style={{
                background: pathname === path ? 'var(--accent)' : 'transparent',
                color: pathname === path ? 'var(--bg-deep)' : 'var(--text-muted)',
              }}
            >
              {label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2 text-xs shrink-0 font-body" style={{ color: 'var(--text-muted)' }}>
          <span className="hidden md:inline font-medium" style={{ color: 'var(--accent)' }}>{name}</span>
          <span className="px-2 py-0.5 rounded-md text-xs font-semibold" style={{ background: 'var(--accent)', color: 'var(--bg-deep)', fontFamily: 'var(--pixel-font)', fontSize: 8 }}>
            Lv.{level} {levelInfo.title}
          </span>
          {accuracy > 0 && (
            <span style={{ color: 'var(--text-muted)' }}>{formatAccuracy(accuracy)}</span>
          )}
        </div>
      </div>
    </header>
  );
}
