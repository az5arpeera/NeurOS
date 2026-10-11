// NeurOS JavaScript client (D-014): receive intent events from the local engine.
//
//   import { NeurOSClient } from "./neuros-client.js";
//   const client = new NeurOSClient("ws://127.0.0.1:8765");
//   client.on("intent", (e) => { if (e.command !== "HOLD") move(e.command); });
//   client.on("status", (e) => console.log("engine is", e.state));
//   client.connect();
//
// Messages follow schema "neuros.intent.v0" (architecture §7.3):
//   hello       - on connect: the enabled commands, the engine's state, the requests it accepts
//   intent      - one per decision (8 per second): command or "HOLD" + hold_reason, probs,
//                 quality; calibration.level is null (zero calibration) or "quick"
//   status      - engine state changes: settling, adapting, steady, degraded, disconnected
//   calibration - the optional guided calibration (D-040): "started", then per trial "ready",
//                 "cue" (show the arrow for e.command), "imagine", "rest"; then "training" and
//                 "done" or "failed" (e.message says why); also "cancelled" and "reset"
//   telemetry   - opt-in (neuros run --telemetry), for displays: with each decision, the newest
//                 cleaned samples of every channel (e.samples, at e.sfreq), each channel's RMS,
//                 the gate's bad channels and the warm-up's seconds_of_data. hello.info then
//                 carries labels for a display (source, channels, threshold, steady_s, ...).
//
// Requests (only those the engine listed in hello.requests):
//   client.calibrate()          start the ~3-minute guided calibration
//   client.cancelCalibration()  stop it; the model in use stays
//   client.resetCalibration()   back to zero calibration
//   client.mark("LEFT")         a ground-truth marker (e.g. the door a game asked for), saved
//                               with the session when the engine records it (neuros run --record)
//
// The client reconnects by itself (with a growing pause, capped) when the engine restarts, and
// never throws on a malformed message: it reports it through the "error" listeners instead.
// No dependencies; works in any browser and in Node >= 22 (global WebSocket).

export const SCHEMA = "neuros.intent.v0";

export class NeurOSClient {
  /**
   * @param {string} url  the engine's WebSocket, e.g. "ws://127.0.0.1:8765"
   * @param {{reconnect?: boolean, maxDelayMs?: number}} [options]
   */
  constructor(url = "ws://127.0.0.1:8765", options = {}) {
    if (typeof url !== "string" || !/^wss?:\/\//.test(url)) {
      throw new TypeError(`NeurOSClient needs a ws:// or wss:// URL, got ${url}`);
    }
    this.url = url;
    this.reconnect = options.reconnect ?? true;
    this.maxDelayMs = options.maxDelayMs ?? 5000;
    this.listeners = {
      hello: [], intent: [], status: [], calibration: [], telemetry: [],
      open: [], close: [], error: [],
    };
    this.socket = null;
    this.delayMs = 250;
    this.closedByUser = false;
    this.commands = [];
    this.requests = [];
    this.state = "disconnected";
  }

  /**
   * Register a listener for "hello", "intent", "status", "calibration", "telemetry", "open",
   * "close" or "error".
   */
  on(type, listener) {
    if (!(type in this.listeners)) throw new TypeError(`unknown event type ${type}`);
    if (typeof listener !== "function") throw new TypeError("listener must be a function");
    this.listeners[type].push(listener);
    return this;
  }

  connect() {
    this.closedByUser = false;
    const socket = new WebSocket(this.url);
    this.socket = socket;
    socket.onopen = () => {
      this.delayMs = 250;
      this.emit("open", { url: this.url });
    };
    socket.onmessage = (message) => this.receive(message.data);
    socket.onerror = () => this.emit("error", { message: `connection to ${this.url} failed` });
    socket.onclose = () => {
      this.state = "disconnected";
      this.emit("close", { url: this.url });
      if (this.reconnect && !this.closedByUser) {
        setTimeout(() => this.connect(), this.delayMs);
        this.delayMs = Math.min(this.delayMs * 2, this.maxDelayMs);
      }
    };
    return this;
  }

  close() {
    this.closedByUser = true;
    if (this.socket) this.socket.close();
  }

  /**
   * Send a request the engine accepts (listed in hello.requests). Returns false, without
   * sending, when the engine isn't connected or doesn't accept it.
   */
  request(name) {
    if (!this.requests.includes(name)) return false;
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify({ type: "request", schema: SCHEMA, request: name }));
    return true;
  }

  calibrate() { return this.request("calibrate"); }
  cancelCalibration() { return this.request("cancel_calibration"); }
  resetCalibration() { return this.request("reset_calibration"); }

  /** A ground-truth marker for a recorded session; label must be one of the commands. */
  mark(label) {
    if (!this.commands.includes(label) || !this.requests.includes("mark")) return false;
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify({ type: "request", schema: SCHEMA, request: "mark", label }));
    return true;
  }

  receive(text) {
    let event;
    try {
      event = JSON.parse(text);
    } catch {
      this.emit("error", { message: "a message wasn't JSON", text });
      return;
    }
    if (!event || event.schema !== SCHEMA || !(event.type in this.listeners)) {
      this.emit("error", { message: "unexpected message", event });
      return;
    }
    if (event.type === "hello") {
      this.commands = Array.isArray(event.commands) ? event.commands : [];
      this.requests = Array.isArray(event.requests) ? event.requests : [];
      this.state = event.state;
    } else if (event.type === "status") {
      this.state = event.state;
    }
    this.emit(event.type, event);
  }

  emit(type, event) {
    for (const listener of this.listeners[type]) {
      try {
        listener(event);
      } catch (error) {
        if (type !== "error") this.emit("error", { message: "a listener threw", error });
      }
    }
  }
}
