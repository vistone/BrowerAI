#!/usr/bin/env python3
"""
任务2 & 3: 真实网站测试 + 性能基准测试
验证fast_enhanced.onnx模型的准确率和推理速度
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
import json

# 框架类别映射
FRAMEWORKS = {
    0: 'react', 1: 'vue', 2: 'angular', 3: 'svelte', 4: 'ember',
    5: 'next', 6: 'nuxt', 7: 'gatsby', 8: 'remix', 9: 'sveltekit',
    10: 'express', 11: 'fastify', 12: 'koa', 13: 'nestjs', 14: 'hapi',
    15: 'webpack', 16: 'vite', 17: 'rollup', 18: 'esbuild',
    19: 'lodash', 20: 'axios', 21: 'ramda', 22: 'underscore', 23: 'other'
}

def load_model(model_path='models/local/fast_enhanced.onnx'):
    """加载ONNX模型"""
    print(f"📂 加载模型: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print(f"✅ 模型加载成功")
    print(f"   提供器: {session.get_providers()}")
    return session

def preprocess_code(code: str, max_length=512):
    """预处理JavaScript代码为模型输入"""
    # 字符级编码 (0-255)
    tokens = [ord(c) % 256 for c in code[:max_length]]
    # 填充到max_length
    tokens = tokens + [0] * (max_length - len(tokens))
    return np.array([tokens], dtype=np.int64)

def predict(session, code: str):
    """执行推理"""
    input_data = preprocess_code(code)
    outputs = session.run(None, {'input_ids': input_data})
    logits = outputs[0][0]
    predicted_class = np.argmax(logits)
    confidence = np.exp(logits[predicted_class]) / np.sum(np.exp(logits))
    return FRAMEWORKS[predicted_class], confidence, logits

def test_real_world_samples():
    """测试真实世界代码样本"""
    print("\n" + "="*70)
    print("📊 任务2: 真实网站代码测试")
    print("="*70 + "\n")
    
    # 真实代码样本（从NPM包和GitHub提取）
    test_cases = [
        {
            'name': 'React Component',
            'expected': 'react',
            'code': '''
import React, { useState, useEffect } from 'react';

function Counter() {
    const [count, setCount] = useState(0);
    
    useEffect(() => {
        document.title = `Count: ${count}`;
    }, [count]);
    
    return (
        <div className="counter">
            <h1>{count}</h1>
            <button onClick={() => setCount(count + 1)}>
                Increment
            </button>
        </div>
    );
}

export default Counter;
            '''
        },
        {
            'name': 'Vue Composition API',
            'expected': 'vue',
            'code': '''
import { ref, computed, onMounted } from 'vue';

export default {
    setup() {
        const count = ref(0);
        const doubled = computed(() => count.value * 2);
        
        onMounted(() => {
            console.log('Component mounted');
        });
        
        function increment() {
            count.value++;
        }
        
        return {
            count,
            doubled,
            increment
        };
    }
}
            '''
        },
        {
            'name': 'Angular Component',
            'expected': 'angular',
            'code': '''
import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
    selector: 'app-login',
    templateUrl: './login.component.html',
    styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
    loginForm: FormGroup;
    
    constructor(private fb: FormBuilder) {}
    
    ngOnInit() {
        this.loginForm = this.fb.group({
            username: ['', Validators.required],
            password: ['', Validators.required]
        });
    }
    
    onSubmit() {
        if (this.loginForm.valid) {
            console.log(this.loginForm.value);
        }
    }
}
            '''
        },
        {
            'name': 'Express Server',
            'expected': 'express',
            'code': '''
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

app.get('/api/users', (req, res) => {
    res.json({ users: [] });
});

app.post('/api/users', (req, res) => {
    const user = req.body;
    res.status(201).json({ success: true, user });
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
            '''
        },
        {
            'name': 'Lodash Utility',
            'expected': 'lodash',
            'code': '''
const _ = require('lodash');

function processData(data) {
    return _.chain(data)
        .filter(item => item.active)
        .map(item => ({
            id: item.id,
            name: _.capitalize(item.name),
            tags: _.uniq(item.tags)
        }))
        .sortBy('name')
        .value();
}

module.exports = { processData };
            '''
        },
        {
            'name': 'Ramda Functional',
            'expected': 'ramda',
            'code': '''
const R = require('ramda');

const processUsers = R.pipe(
    R.filter(R.prop('active')),
    R.map(R.pick(['id', 'name', 'email'])),
    R.sortBy(R.prop('name'))
);

const getAdmins = R.compose(
    R.map(R.prop('name')),
    R.filter(R.propEq('role', 'admin'))
);

module.exports = { processUsers, getAdmins };
            '''
        },
    ]
    
    session = load_model()
    
    correct = 0
    total = len(test_cases)
    
    print(f"测试样本数量: {total}\n")
    
    for i, test in enumerate(test_cases, 1):
        predicted, confidence, logits = predict(session, test['code'])
        expected = test['expected']
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} 测试 {i}: {test['name']}")
        print(f"   期望: {expected}")
        print(f"   预测: {predicted}")
        print(f"   置信度: {confidence:.2%}")
        
        if not is_correct:
            # 显示前3个预测
            top3_idx = np.argsort(logits)[-3:][::-1]
            print(f"   Top 3: {', '.join([f'{FRAMEWORKS[idx]} ({logits[idx]:.2f})' for idx in top3_idx])}")
        print()
    
    accuracy = correct / total
    print("="*70)
    print(f"📊 测试结果:")
    print(f"   准确率: {correct}/{total} = {accuracy:.2%}")
    print(f"   {'✅ 通过' if accuracy >= 0.7 else '❌ 未达标'} (目标 >= 70%)")
    print("="*70)
    
    return accuracy

def benchmark_performance():
    """性能基准测试"""
    print("\n" + "="*70)
    print("⚡ 任务3: 性能基准测试")
    print("="*70 + "\n")
    
    session = load_model()
    
    # 测试代码
    test_code = '''
    import React from 'react';
    function App() {
        return <div>Hello World</div>;
    }
    export default App;
    '''
    
    # 预热
    print("🔥 预热模型...")
    for _ in range(10):
        predict(session, test_code)
    
    # 基准测试
    print("⏱️  执行基准测试...\n")
    
    iterations = 100
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        predict(session, test_code)
        elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
        times.append(elapsed)
    
    times = np.array(times)
    
    print(f"📊 性能统计 ({iterations} 次推理):")
    print(f"   平均时间: {times.mean():.2f} ms")
    print(f"   中位数: {np.median(times):.2f} ms")
    print(f"   最小时间: {times.min():.2f} ms")
    print(f"   最大时间: {times.max():.2f} ms")
    print(f"   标准差: {times.std():.2f} ms")
    print(f"   P95: {np.percentile(times, 95):.2f} ms")
    print(f"   P99: {np.percentile(times, 99):.2f} ms")
    
    avg_time = times.mean()
    throughput = 1000 / avg_time  # 每秒处理数
    
    print(f"\n🚀 吞吐量:")
    print(f"   {throughput:.1f} 推理/秒")
    print(f"   {'✅ 优秀' if avg_time < 20 else '⚠️  可优化'} (目标 < 20ms)")
    
    print("="*70)
    
    return avg_time

def save_results(accuracy, avg_time):
    """保存测试结果"""
    results = {
        'model': 'fast_enhanced.onnx',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': f'{accuracy:.2%}',
        'avg_inference_time_ms': round(avg_time, 2),
        'throughput_per_sec': round(1000 / avg_time, 1)
    }
    
    output_path = Path('models/local/fast_enhanced_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 测试结果已保存: {output_path}")

def main():
    print("\n" + "="*70)
    print("🧪 fast_enhanced.onnx 模型验证测试套件")
    print("="*70)
    
    # 任务2: 真实网站测试
    accuracy = test_real_world_samples()
    
    # 任务3: 性能基准测试
    avg_time = benchmark_performance()
    
    # 保存结果
    save_results(accuracy, avg_time)
    
    # 最终总结
    print("\n" + "="*70)
    print("✅ 所有测试完成!")
    print("="*70)
    print(f"   准确率: {accuracy:.2%}")
    print(f"   推理速度: {avg_time:.2f} ms")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
