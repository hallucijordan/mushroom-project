import { motion } from 'framer-motion';

interface Props {
  onReveal: (label: 'edible' | 'poisonous') => void;
  disabled?: boolean;
}

export function OracleReveal({ onReveal, disabled }: Props) {
  return (
    <div className="w-full">
      <p className="text-xs text-gray-400 uppercase tracking-wide text-center mb-3">
        What is it really?
      </p>
      <div className="grid grid-cols-2 gap-3">
        <motion.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          disabled={disabled}
          onClick={() => onReveal('edible')}
          className="flex flex-col items-center gap-1 bg-green-50 hover:bg-green-100 disabled:opacity-50 border border-green-200 rounded-2xl py-4 px-3 font-semibold text-green-700 transition-colors"
        >
          <span className="text-3xl">✅</span>
          <span>Edible</span>
        </motion.button>

        <motion.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          disabled={disabled}
          onClick={() => onReveal('poisonous')}
          className="flex flex-col items-center gap-1 bg-red-50 hover:bg-red-100 disabled:opacity-50 border border-red-200 rounded-2xl py-4 px-3 font-semibold text-red-700 transition-colors"
        >
          <span className="text-3xl">☠️</span>
          <span>Poisonous</span>
        </motion.button>
      </div>
    </div>
  );
}
