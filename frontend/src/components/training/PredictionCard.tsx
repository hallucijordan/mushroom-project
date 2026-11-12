import { motion } from 'framer-motion';
import type { Prediction } from '../../types/training';
import { formatConfidence } from '../../utils/formatters';

interface Props {
  prediction: Prediction;
}

export function PredictionCard({ prediction }: Props) {
  const isPoison = prediction.label === 'poisonous';
  const confPercent = Math.round(prediction.confidence * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-2xl border shadow-sm p-4 w-full"
    >
      <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Cat's Prediction</p>

      <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-lg font-bold mb-3 ${
        isPoison ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-green-50 text-green-700 border border-green-200'
      }`}>
        <span>{isPoison ? '☠️' : '✅'}</span>
        <span className="capitalize">{prediction.label}</span>
      </div>

      {/* Confidence bar */}
      <div className="mb-3">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Confidence</span>
          <span className="font-medium text-gray-600">{formatConfidence(prediction.confidence)}</span>
        </div>
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${isPoison ? 'bg-red-400' : 'bg-green-400'}`}
            initial={{ width: 0 }}
            animate={{ width: `${confPercent}%` }}
            transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
          />
        </div>
      </div>

      {/* Dual probability */}
      <div className="grid grid-cols-2 gap-2 text-center text-xs">
        <div className="bg-green-50 rounded-lg p-2">
          <p className="text-gray-400">Edible</p>
          <p className="font-bold text-green-600">{formatConfidence(prediction.p_edible)}</p>
        </div>
        <div className="bg-red-50 rounded-lg p-2">
          <p className="text-gray-400">Poisonous</p>
          <p className="font-bold text-red-600">{formatConfidence(prediction.p_poisonous)}</p>
        </div>
      </div>
    </motion.div>
  );
}
