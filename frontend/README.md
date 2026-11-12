# Frontend (UI prototype)

React + TypeScript + Vite UI for the active-learning web app. See the main README, Part 3.

**Status: prototype.** The pages are not fully connected to the backend yet. In dev mode the UI uses
simulated data and shows a banner saying so. Known gaps:

- The strategy comparison page calls `POST /api/training/compare`, which the backend does not have yet.
- The training step and learning-curve responses use different field names and shapes in the backend
  (snake_case) and here (camelCase).
- The photo upload sends the field `file`; the backend expects `image`.

```bash
npm install
npm run dev                          # http://localhost:5173, simulated data
VITE_USE_MOCK=false npm run dev      # call the backend at localhost:8000 instead
```

Outside mock mode a failed API call is an error. The UI never switches to simulated data on its own.
