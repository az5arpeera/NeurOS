// NeurOS JavaScript client (D-014): receive intent events from the local engine.
//
//   import { NeurOSClient } from "./neuros-client.js";
//   const client = new NeurOSClient("ws://127.0.0.1:8765");
//   client.on("intent", (e) => { if (e.command !== "HOLD") move(e.command); });
//   client.on("status", (e) => console.log("engine is", e.state));
//   client.connect();
//
// Messages follow schema "neuros.intent.v0" (architecture §7.3):
//   hello  - on connect: the enabled commands and the engine's current state
//   intent - one per decision (8 per second): command or "HOLD" + hold_reason, probs, quality
//   status - engine state changes: settling, adapting, steady, degraded, disconnected
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
    this.listeners = { hello: [], intent: [], status: [], open: [], close: [], error: [] };
    this.socket = null;
    this.delayMs = 250;
    this.closedByUser = false;
    this.commands = [];
    this.state = "disconnected";
  }

  /** Register a listener for "hello", "intent", "status", "open", "close" or "error". */
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
