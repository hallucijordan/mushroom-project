interface Props { size?: 'sm' | 'md' | 'lg'; label?: string; }

export function LoadingSpinner({ size = 'md', label }: Props) {
  const sizes = { sm: 'w-4 h-4', md: 'w-8 h-8', lg: 'w-12 h-12' };
  return (
    <div className="flex flex-col items-center gap-2">
      <div className={`${sizes[size]} border-3 border-amber-200 border-t-amber-500 rounded-full animate-spin`} />
      {label && <p className="text-sm font-body" style={{ color: 'var(--text-muted)' }}>{label}</p>}
    </div>
  );
}
