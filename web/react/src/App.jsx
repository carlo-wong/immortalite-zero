import { useCallback, useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessground } from "chessground";

const API = "/analyze";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

function toDests(chess) {
  const dests = new Map();
  for (const m of chess.moves({ verbose: true })) {
    if (!dests.has(m.from)) dests.set(m.from, []);
    dests.get(m.from).push(m.to);
  }
  return dests;
}

function toWhiteCp(cp, turn) {
  return turn === "w" ? cp : -cp;
}

function formatCp(cp) {
  const v = cp / 100;
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}

export default function App() {
  const boardRef = useRef(null);
  const cgRef = useRef(null);
  const [chess] = useState(() => new Chess());
  const [fen, setFen] = useState(chess.fen());
  const [analysis, setAnalysis] = useState(null);
  const [status, setStatus] = useState("");
  const [health, setHealth] = useState(null);
  const seqRef = useRef(0);

  const syncBoard = useCallback(() => {
    if (!cgRef.current) return;
    const turn = chess.turn();
    cgRef.current.set({
      fen: chess.fen(),
      turnColor: turn,
      check: chess.inCheck(),
      movable: {
        color: chess.isGameOver() ? undefined : turn,
        dests: chess.isGameOver() ? undefined : toDests(chess),
      },
    });
    setFen(chess.fen());
  }, [chess]);

  useEffect(() => {
    if (!boardRef.current) return;
    cgRef.current = Chessground(boardRef.current, {
      fen: chess.fen(),
      animation: { enabled: true },
      movable: {
        free: false,
        events: {
          after: (orig, dest) => {
            const promo = chess.get(orig)?.type === "p" &&
              ((chess.turn() === "w" && dest[1] === "8") || (chess.turn() === "b" && dest[1] === "1"));
            const mv = chess.move({ from: orig, to: dest, promotion: promo ? "q" : undefined });
            if (mv) syncBoard();
          },
        },
      },
    });
    syncBoard();
    return () => cgRef.current?.destroy();
  }, [chess, syncBoard]);

  const runAnalyze = useCallback(async () => {
    if (chess.isGameOver()) {
      setStatus("Game over.");
      setAnalysis(null);
      return;
    }
    const seq = ++seqRef.current;
    setStatus("Analyzing…");
    try {
      const res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: chess.fen(), multipv: 5, simulations: 200 }),
      });
      if (seq !== seqRef.current) return;
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (seq !== seqRef.current) return;
      setAnalysis(data);
      setStatus(`Done (${data.elapsed_ms ?? "?"} ms)`);

      const shapes = [];
      if (data.best_move) {
        shapes.push({
          orig: data.best_move.slice(0, 2),
          dest: data.best_move.slice(2, 4),
          brush: "blue",
        });
      }
      cgRef.current?.setAutoShapes(shapes.map(s => ({ ...s, opacity: 0.7 })));
    } catch {
      if (seq !== seqRef.current) return;
      setStatus("Backend offline");
    }
  }, [chess]);

  useEffect(() => {
    fetch("/health").then(r => r.json()).then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    const t = setTimeout(runAnalyze, 300);
    return () => clearTimeout(t);
  }, [fen, runAnalyze]);

  return (
    <div style={{ background: "#161512", color: "#bababa", minHeight: "100vh", padding: 24, fontFamily: "system-ui" }}>
      <h1 style={{ color: "#e8e6e3", marginTop: 0 }}>Immortalite Zero — React + Chessground</h1>
      <p style={{ fontSize: 13 }}>
        Experimental Chessground UI. Feature parity scaffold — use <a href="/app/" style={{ color: "#759900" }}>/app/</a> for full GUI.
      </p>
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <div ref={boardRef} style={{ width: 480, height: 480 }} />
        <div style={{ width: 300 }}>
          <button onClick={runAnalyze} style={{ background: "#759900", color: "#fff", border: "none", padding: "8px 14px", borderRadius: 4, cursor: "pointer" }}>
            Analyze
          </button>
          <button onClick={() => { chess.reset(); syncBoard(); }} style={{ marginLeft: 8, background: "#3a3835", color: "#fff", border: "none", padding: "8px 14px", borderRadius: 4, cursor: "pointer" }}>
            Reset
          </button>
          <p style={{ fontSize: 12 }}>{status}</p>
          {analysis && (
            <>
              <p style={{ fontSize: 14, color: "#e8e6e3" }}>Eval: {formatCp(toWhiteCp(analysis.eval_cp, chess.turn()))}</p>
              {analysis.lines?.map((line, i) => (
                <div key={i} style={{ background: "#1f1e1b", padding: 8, marginBottom: 6, borderRadius: 4, fontSize: 13 }}>
                  <strong>{line.move}</strong> {formatCp(toWhiteCp(line.eval_cp, chess.turn()))}
                </div>
              ))}
            </>
          )}
          <p style={{ fontSize: 11, color: "#666", marginTop: 16 }}>
            {health
              ? `Checkpoint: ${health.trained ? health.checkpoint.split(/[/\\]/).pop() : "untrained"} · enc v${health.encoding_version}`
              : "Server offline"}
          </p>
        </div>
      </div>
    </div>
  );
}
