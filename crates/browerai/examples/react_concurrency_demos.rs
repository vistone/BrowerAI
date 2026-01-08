//! React 并发特性小实验示例（startTransition / useTransition / useDeferredValue / Suspense）
//! 说明：这是一个“教程式”示例，直接在终端输出三段可复制的 React 18 代码片段与观察要点。
//! 运行：
//!   cargo run --example react_concurrency_demos

fn main() {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║ React 18 并发特性小实验 (CLI 教程)                  ║");
    println!("╚════════════════════════════════════════════════════╝\n");

    section_filter_demo();
    section_deferred_demo();
    section_suspense_demo();

    println!("\n✅ 复制以上代码到你的 React 18 项目，即可直接实验。");
    println!("   建议搭配 React DevTools Profiler 观察渲染次数与 pending 状态。");
}

fn section_filter_demo() {
    let code = r#"import { useMemo, useState, useTransition } from 'react';

const bigList = Array.from({ length: 20000 }, (_, i) => `Item ${i}`);

export default function FilterDemo() {
  const [text, setText] = useState('');
  const [isPending, startTransition] = useTransition();

  const handleChange = (e) => {
    const value = e.target.value;
    // 将重计算放到低优先级，保持输入流畅
    startTransition(() => setText(value));
  };

  const filtered = useMemo(() => {
    const lower = text.toLowerCase();
    return bigList.filter((item) => item.toLowerCase().includes(lower));
  }, [text]);

  return (
    <div>
      <input placeholder=\"filter...\" onChange={handleChange} />
      {isPending && <p>⌛ 正在计算...</p>}
      <div style={{ maxHeight: 200, overflow: 'auto' }}>
        {filtered.slice(0, 200).map((item) => (
          <div key={item}>{item}</div>
        ))}
      </div>
    </div>
  );
}
"#;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎣 实验 1：输入 + 大列表过滤（startTransition/useTransition）");
    println!("要点：将昂贵的过滤放入 startTransition，保持输入不卡顿；isPending 可做加载提示。\n");
    println!("代码：\n{}", code);
}

fn section_deferred_demo() {
    let code = r#"import { useDeferredValue, useMemo, useState } from 'react';

const bigList = Array.from({ length: 20000 }, (_, i) => `Row ${i}`);

export default function DeferredDemo() {
  const [text, setText] = useState('');
  const deferredText = useDeferredValue(text); // 延迟版本，减少每次击键的重渲染

  const filtered = useMemo(() => {
    const lower = deferredText.toLowerCase();
    return bigList.filter((x) => x.toLowerCase().includes(lower));
  }, [deferredText]);

  return (
    <div>
      <input placeholder=\"filter...\" value={text} onChange={(e) => setText(e.target.value)} />
      <div style={{ maxHeight: 200, overflow: 'auto' }}>
        {filtered.slice(0, 200).map((x) => (
          <div key={x}>{x}</div>
        ))}
      </div>
    </div>
  );
}
"#;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("⏳ 实验 2：延迟值渲染（useDeferredValue）");
    println!("要点：输入框即时响应，列表过滤使用延迟值，避免每次击键都触发昂贵计算。\n");
    println!("代码：\n{}", code);
}

fn section_suspense_demo() {
    let code = r#"import React, { Suspense, lazy, useState } from 'react';

const SlowComp = lazy(() => new Promise((res) => {
  setTimeout(() => res(import('./SlowComp')), 1500); // 模拟 1.5s 延迟
}));

export default function SuspenseDemo() {
  const [show, setShow] = useState(false);
  return (
    <div>
      <button onClick={() => setShow((v) => !v)}>Toggle</button>
      <Suspense fallback={<p>⌛ 加载中...</p>}>
        {show && <SlowComp />}
      </Suspense>
    </div>
  );
}

// SlowComp.jsx
export default function SlowComp() {
  return <div>✅ 异步组件加载完成</div>;
}
"#;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🌀 实验 3：Suspense + lazy（异步组件与 fallback）");
    println!("要点：切换时先展示 fallback，占位 1.5s 后加载完成；可替换为真实网络请求。\n");
    println!("代码：\n{}", code);
}
