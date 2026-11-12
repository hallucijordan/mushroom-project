export type CatMood =
  | 'idle'
  | 'thinking'
  | 'happy'
  | 'sick'
  | 'proud'
  | 'curious'
  | 'sleeping'
  | 'confused'
  | 'celebrating';

export type CatLevel = 0 | 1 | 2 | 3 | 4 | 5;

export interface CatLevelInfo {
  level: CatLevel;
  title: string;
  minAccuracy: number;
  maxAccuracy: number;
  accessory: string;
  unlocksFeature: string;
}

export const CAT_LEVELS: CatLevelInfo[] = [
  { level: 0, title: 'Clueless Kitten',  minAccuracy: 0,    maxAccuracy: 0.59, accessory: 'None',                      unlocksFeature: 'Training begins' },
  { level: 1, title: 'Curious Nibbler',  minAccuracy: 0.60, maxAccuracy: 0.74, accessory: 'Magnifying glass',           unlocksFeature: 'Strategy picker' },
  { level: 2, title: 'Spore Spotter',    minAccuracy: 0.75, maxAccuracy: 0.84, accessory: 'Field notebook',             unlocksFeature: 'Learning curve chart' },
  { level: 3, title: 'Gill Inspector',   minAccuracy: 0.85, maxAccuracy: 0.89, accessory: 'Lab coat',                   unlocksFeature: 'Expert mode' },
  { level: 4, title: 'Myco Apprentice',  minAccuracy: 0.90, maxAccuracy: 0.94, accessory: 'ID chart scroll',            unlocksFeature: 'Strategy comparison' },
  { level: 5, title: 'Fungal Master',    minAccuracy: 0.95, maxAccuracy: 1.00, accessory: 'Diploma + golden mushroom pin', unlocksFeature: 'Full dashboard' },
];

export const CAT_LEVEL_UP_DIALOGUES: Record<string, string> = {
  '0→1': "I can tell mushrooms apart from rocks now! That's... progress!",
  '1→2': "I got a notebook! Time to take proper notes on these specimens.",
  '2→3': "They gave me a LAB COAT! I must be getting serious about this.",
  '3→4': "I can identify gill patterns in my sleep now. Bring on the tough ones!",
  '4→5': "I... I did it. I'm a Fungal Master. *adjusts reading glasses* All those mushrooms were worth it.",
};
