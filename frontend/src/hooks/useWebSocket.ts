import { useEffect, useRef, useCallback } from 'react';

type WSMessage =
  | { type: 'retrain_progress'; progress: number }
  | { type: 'retrain_complete'; new_accuracy: number; cat_reaction: string }
  | { type: 'level_up'; new_level: number; dialogue: string };

export function useWebSocket(
  sessionId: string | null,
  onMessage: (msg: WSMessage) => void
) {
  const wsRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (!sessionId) return;
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/training/${sessionId}`);
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as WSMessage;
        onMessageRef.current(msg);
      } catch { /* ignore parse errors */ }
    };
    ws.onclose = () => {
      // Reconnect after 3s if still mounted
      setTimeout(connect, 3000);
    };
    wsRef.current = ws;
  }, [sessionId]);

  useEffect(() => {
    connect();
    return () => { wsRef.current?.close(); };
  }, [connect]);
}
