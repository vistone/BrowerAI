// simple counter module
export function createCounter(initial = 0) {
  let value = initial;
  return {
    inc(step = 1) {
      value += step;
      return value;
    },
    dec(step = 1) {
      value -= step;
      return value;
    },
    get() {
      return value;
    }
  };
}
