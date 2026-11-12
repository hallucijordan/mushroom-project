import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  text: string;
  speakerName?: string;
  typewriterSpeed?: number;
}

export function CatDialogue({ text, speakerName = 'Maodie', typewriterSpeed = 30 }: Props) {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setDisplayed('');
    setDone(false);
    let i = 0;
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(intervalRef.current!);
        setDone(true);
      }
    }, typewriterSpeed);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [text, typewriterSpeed]);

  return (
    <AnimatePresence>
      <motion.div
        key={text}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative rounded-xl rounded-bl-sm border px-4 py-3 max-w-sm"
        style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
      >
        <p className="mb-1" style={{ fontFamily: 'var(--pixel-font)', fontSize: 8, color: 'var(--accent)' }}>{speakerName}</p>
        <p className="text-sm leading-relaxed min-h-[3rem] font-body" style={{ color: 'var(--text)' }}>
          {displayed}
          {!done && <span className="inline-block w-0.5 h-4 ml-0.5 animate-pulse" style={{ background: 'var(--accent)' }} />}
        </p>
        {/* Speech bubble tail */}
        <div className="absolute -bottom-3 left-4 w-0 h-0 border-l-8 border-r-8 border-t-8 border-l-transparent border-r-transparent" style={{ borderTopColor: 'var(--border)' }} />
      </motion.div>
    </AnimatePresence>
  );
}
