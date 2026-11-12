import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Header } from './components/shared/Header';
import { Footer } from './components/shared/Footer';
import { HomePage } from './pages/HomePage';
import { TrainingPage } from './pages/TrainingPage';
import { ExpertPage } from './pages/ExpertPage';
import { JourneyPage } from './pages/JourneyPage';
import { ComparisonPage } from './pages/ComparisonPage';
import { isMockMode } from './services/api';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-[var(--bg-deep)]">
        {isMockMode() && (
          <div className="text-center text-xs py-1" style={{ background: 'var(--accent)', color: 'var(--bg-deep)' }}>
            UI prototype: all numbers on this page are simulated, not results from the model.
          </div>
        )}
        <Header />
        <main className="flex-1">
          <Routes>
            <Route path="/"           element={<HomePage />} />
            <Route path="/training"   element={<TrainingPage />} />
            <Route path="/expert"     element={<ExpertPage />} />
            <Route path="/journey"    element={<JourneyPage />} />
            <Route path="/comparison" element={<ComparisonPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
