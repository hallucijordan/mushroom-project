import { useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function MushroomUploader({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const handle = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    onFile(file);
  };

  return (
    <div className="w-full">
      <motion.div
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
        style={{
          borderColor: dragging ? 'var(--accent)' : 'var(--border)',
          background: dragging ? 'rgba(226,163,70,0.08)' : 'var(--bg-raised)',
        }}
        whileTap={{ scale: 0.98 }}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files[0];
          if (file) handle(file);
        }}
      >
        {preview ? (
          <img src={preview} alt="Mushroom preview" className="mx-auto max-h-40 rounded-xl object-cover" />
        ) : (
          <div>
            <p className="text-4xl mb-2">📷</p>
            <p className="text-sm font-body" style={{ color: 'var(--text-muted)' }}>Upload a mushroom photo</p>
            <p className="text-xs mt-1 font-body" style={{ color: 'var(--text-muted)' }}>Drag & drop or click to browse</p>
          </div>
        )}
      </motion.div>

      {preview && (
        <button
          className="mt-2 text-xs w-full text-center font-body"
          style={{ color: 'var(--text-muted)' }}
          onClick={() => { setPreview(null); }}
        >
          Clear photo
        </button>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handle(f); }}
      />
    </div>
  );
}
