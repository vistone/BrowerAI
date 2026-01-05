// event emitter
class Emitter {
  constructor() {
    this.handlers = {};
  }
  on(event, fn) {
    (this.handlers[event] ||= []).push(fn);
  }
  emit(event, payload) {
    (this.handlers[event] || []).forEach((fn) => fn(payload));
  }
}

export function createEmitter() {
  return new Emitter();
}
