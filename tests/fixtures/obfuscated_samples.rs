/// Week 4 Phase 2: 混淆代码样本库
/// 
/// 包含 50+ 个代表性的 JavaScript 代码样本，涵盖 8 种混淆技术

/// 1. 控制流扁平化样本 (5个)
pub const CONTROL_FLOW_SAMPLES: &[&str] = &[
    // 样本 1: 基础 switch-case 扁平化
    r#"
var state = 0;
while (true) {
    switch(state) {
        case 0: console.log('a'); state = 1; break;
        case 1: console.log('b'); state = 2; break;
        case 2: console.log('c'); return;
    }
}
"#,
    // 样本 2: 复杂控制流
    r#"
var x = Math.random();
switch(Math.floor(x * 10)) {
    case 0: func1(); break;
    case 1: func2(); break;
    case 2: func3(); break;
    case 3: func4(); break;
    default: funcDefault();
}
"#,
    // 样本 3: 嵌套 switch
    r#"
switch(a) {
    case 1: 
        switch(b) {
            case 10: x(); break;
            case 20: y(); break;
        }
        break;
    case 2: z(); break;
}
"#,
    // 样本 4: goto 风格
    r#"
var label = 0;
goto_loop: while(true) {
    if (label === 0) { step1(); label = 1; continue goto_loop; }
    if (label === 1) { step2(); label = 2; continue goto_loop; }
    if (label === 2) { break goto_loop; }
}
"#,
    // 样本 5: 深度嵌套
    r#"
if (a) {
    if (b) {
        if (c) {
            if (d) {
                if (e) {
                    deepFunction();
                }
            }
        }
    }
}
"#,
];

/// 2. 字符串编码样本 (5个)
pub const STRING_ENCODING_SAMPLES: &[&str] = &[
    // 样本 1: Hex 编码
    r#"
var secret = "\x48\x65\x6c\x6c\x6f";
var pass = "\x70\x61\x73\x73\x77\x6f\x72\x64";
console.log(secret);
"#,
    // 样本 2: Unicode 编码
    r#"
var msg = "\u0048\u0065\u006c\u006c\u006f";
var key = "\u006b\u0065\u0079";
"#,
    // 样本 3: Base64 解码
    r#"
var encoded = "SGVsbG8gV29ybGQ=";
var decoded = atob(encoded);
console.log(decoded);
"#,
    // 样本 4: 混合编码
    r#"
var a = "\x48\u0065\u006c\x6c\x6f";
var b = atob("V29ybGQ=");
var result = a + " " + b;
"#,
    // 样本 5: 字符串拼接混淆
    r#"
var s = String.fromCharCode(72, 101, 108, 108, 111);
var t = String.fromCharCode(87, 111, 114, 108, 100);
"#,
];

/// 3. 死代码注入样本 (5个)
pub const DEAD_CODE_SAMPLES: &[&str] = &[
    // 样本 1: 永假条件
    r#"
function test() {
    var x = 1;
    if (false) {
        deadCode1();
        deadCode2();
        deadCode3();
    }
    return x + 1;
}
"#,
    // 样本 2: !0 条件
    r#"
var result = 0;
if (!0) {
    neverExecuted();
}
if (!!1) {
    alwaysExecuted();
}
"#,
    // 样本 3: 不可达代码
    r#"
function unreachable() {
    return 42;
    console.log("never printed");
    var x = 100;
}
"#,
    // 样本 4: 未使用变量
    r#"
function unused() {
    var usedVar = 1;
    var unusedVar1 = 2;
    var unusedVar2 = 3;
    var unusedVar3 = 4;
    return usedVar;
}
"#,
    // 样本 5: 重复代码
    r#"
var a = 1 + 1;
var b = 1 + 1;
var c = 1 + 1;
var d = 1 + 1;
"#,
];

/// 4. 变量重命名样本 (5个)
pub const VARIABLE_RENAMING_SAMPLES: &[&str] = &[
    // 样本 1: 短变量名
    r#"
var a = 1;
var b = 2;
var c = a + b;
var d = c * 2;
function f(x) { return x + 1; }
"#,
    // 样本 2: 单字母标识符
    r#"
var x,y,z,w,v,u,t,s,r,q,p,o,n,m;
x = y = z = 0;
function a(b,c,d) { return b+c+d; }
"#,
    // 样本 3: 下划线前缀
    r#"
var _a = 1, _b = 2, _c = 3;
var _0x123 = "secret";
var _0x456 = "key";
"#,
    // 样本 4: Unicode 标识符
    r#"
var α = 1;
var β = 2;
var γ = α + β;
var 中文变量 = "test";
"#,
    // 样本 5: 相似标识符
    r#"
var l = 1;  // lowercase L
var I = 2;  // uppercase i
var O = 3;  // uppercase o
var o = 4;  // lowercase o
"#,
];

/// 5. 代码膨胀样本 (5个)
pub const CODE_BLOAT_SAMPLES: &[&str] = &[
    // 样本 1: 重复语句
    r#"
var x = 1;
x = x + 0;
x = x + 0;
x = x + 0;
x = x * 1;
x = x * 1;
"#,
    // 样本 2: 冗余运算
    r#"
var a = (1 + 2) - 2;
var b = (x * 2) / 2;
var c = x - x + y;
"#,
    // 样本 3: 无用包装
    r#"
function wrapper() {
    return (function() {
        return (function() {
            return 42;
        })();
    })();
}
"#,
    // 样本 4: 长数组/对象
    r#"
var arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
var obj = {a:0,b:0,c:0,d:0,e:0,f:0,g:0,h:0,i:0,j:0};
"#,
    // 样本 5: 重复函数调用
    r#"
foo();foo();foo();foo();foo();
bar();bar();bar();bar();bar();
"#,
];

/// 6. 常量还原样本 (5个)
pub const CONSTANT_RESTORATION_SAMPLES: &[&str] = &[
    // 样本 1: 数学表达式常量
    r#"
var x = 1 + 2 + 3;
var y = 10 * 5 / 2;
var z = Math.pow(2, 10);
"#,
    // 样本 2: 字符串拼接
    r#"
var url = "http://" + "example" + "." + "com";
var path = "/" + "api" + "/" + "v1";
"#,
    // 样本 3: 布尔运算
    r#"
var a = true && true;
var b = false || true;
var c = !false;
"#,
    // 样本 4: 数组操作
    r#"
var arr = [1,2,3];
var first = arr[0];
var len = arr.length;
"#,
    // 样本 5: 对象属性
    r#"
var obj = {x: 10, y: 20};
var sum = obj.x + obj.y;
var prop = obj["x"];
"#,
];

/// 7. API 隐藏样本 (5个)
pub const API_HIDING_SAMPLES: &[&str] = &[
    // 样本 1: 间接调用
    r#"
var f = console.log;
f("hidden");
var g = document.getElementById;
g("id");
"#,
    // 样本 2: 动态属性访问
    r#"
var method = "log";
console[method]("test");
var prop = "length";
arr[prop];
"#,
    // 样本 3: apply/call
    r#"
Function.prototype.call.apply(console.log, [console, "test"]);
setTimeout.call(null, function(){}, 1000);
"#,
    // 样本 4: 别名
    r#"
var $ = document.querySelector;
var _ = Array.prototype.slice;
var __ = Object.keys;
"#,
    // 样本 5: 反射
    r#"
var obj = {};
Object.defineProperty(obj, "hidden", {value: 42});
Reflect.get(obj, "hidden");
"#,
];

/// 8. 动态调用样本 (5个)
pub const DYNAMIC_INVOCATION_SAMPLES: &[&str] = &[
    // 样本 1: eval
    r#"
eval("console.log('dynamic')");
var code = "1 + 1";
var result = eval(code);
"#,
    // 样本 2: Function 构造
    r#"
var fn = new Function("x", "return x + 1");
var result = fn(5);
"#,
    // 样本 3: setTimeout 字符串
    r#"
setTimeout("alert('delayed')", 1000);
setInterval("counter++", 100);
"#,
    // 样本 4: document.write
    r#"
document.write("<script>alert('injected')</script>");
"#,
    // 样本 5: 动态 import
    r#"
var module = "lodash";
import(module).then(lib => lib.default());
"#,
];

/// 混合样本 (10个复杂示例)
pub const MIXED_SAMPLES: &[&str] = &[
    // 混合 1: 控制流 + 字符串编码
    r#"
var state = 0;
while(true) {
    switch(state) {
        case 0: 
            var s = "\x48\x65\x6c\x6c\x6f";
            state = 1; 
            break;
        case 1: 
            console.log(s); 
            return;
    }
}
"#,
    // 混合 2: 死代码 + 变量重命名
    r#"
var a=1,b=2,c=3;
if(false){var d=4,e=5;}
function f(x){return x+a;}
"#,
    // 混合 3: 字符串 + API 隐藏
    r#"
var m = "log";
var s = "\x74\x65\x73\x74";
console[m](atob("SGVsbG8="));
"#,
    // 混合 4: 控制流 + 动态调用
    r#"
var x = Math.random();
switch(Math.floor(x*2)) {
    case 0: eval("func1()"); break;
    case 1: new Function("func2()")(); break;
}
"#,
    // 混合 5: 全技术混合
    r#"
var _0x123 = "\x70\x61\x73\x73";
var state = 0;
if(false){deadCode();}
while(true) {
    switch(state) {
        case 0: 
            var f = console["log"];
            state = 1; 
            break;
        case 1: 
            eval('f(atob("' + btoa(_0x123) + '"))');
            return;
    }
}
"#,
    // 混合 6-10: 真实混淆器输出模拟
    r#"
(function(_0x4d8f,_0x2c71){var _0x5b3a=function(_0x186c){while(--_0x186c){_0x4d8f['push'](_0x4d8f['shift']());}};_0x5b3a(++_0x2c71);}(_0x3f,0x1a4));
"#,
    r#"
var _0x=['secret','key','\x68\x65\x6c\x6c\x6f'];(function(a,b){var c=function(d){while(--d){a['push'](a['shift']());}};c(++b);}(_0x,0x123));
"#,
    r#"
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?module.exports=t():"function"==typeof define&&define.amd?define(t):e.MyLib=t()}(this,function(){return function(){console.log("obfuscated")}});
"#,
    r#"
var a={};!function(){var b=function(c){return c*2};Object.defineProperty(a,"calc",{get:function(){return b}})}();
"#,
    r#"
eval(function(p,a,c,k,e,d){while(c--)if(k[c])p=p.replace(new RegExp('\\b'+c+'\\b','g'),k[c]);return p}('0 1(){2.3("4 5")}',6,6,'function|test|console|log|Hello|World'.split('|'),0,{}));
"#,
];

/// 获取所有样本（50+个）
pub fn get_all_samples() -> Vec<(&'static str, &'static str)> {
    let mut samples = Vec::new();
    
    for (i, sample) in CONTROL_FLOW_SAMPLES.iter().enumerate() {
        samples.push((*sample, "ControlFlowFlattening"));
    }
    for (i, sample) in STRING_ENCODING_SAMPLES.iter().enumerate() {
        samples.push((*sample, "StringEncoding"));
    }
    for (i, sample) in DEAD_CODE_SAMPLES.iter().enumerate() {
        samples.push((*sample, "DeadCodeInjection"));
    }
    for (i, sample) in VARIABLE_RENAMING_SAMPLES.iter().enumerate() {
        samples.push((*sample, "VariableRenaming"));
    }
    for (i, sample) in CODE_BLOAT_SAMPLES.iter().enumerate() {
        samples.push((*sample, "CodeBloat"));
    }
    for (i, sample) in CONSTANT_RESTORATION_SAMPLES.iter().enumerate() {
        samples.push((*sample, "ConstantRestoration"));
    }
    for (i, sample) in API_HIDING_SAMPLES.iter().enumerate() {
        samples.push((*sample, "APIHiding"));
    }
    for (i, sample) in DYNAMIC_INVOCATION_SAMPLES.iter().enumerate() {
        samples.push((*sample, "DynamicInvocation"));
    }
    for (i, sample) in MIXED_SAMPLES.iter().enumerate() {
        samples.push((*sample, "Mixed"));
    }
    
    samples
}

/// 获取样本统计
pub fn get_sample_stats() -> String {
    format!(
        "Total samples: {}\n\
         - Control Flow: {}\n\
         - String Encoding: {}\n\
         - Dead Code: {}\n\
         - Variable Renaming: {}\n\
         - Code Bloat: {}\n\
         - Constant Restoration: {}\n\
         - API Hiding: {}\n\
         - Dynamic Invocation: {}\n\
         - Mixed: {}",
        get_all_samples().len(),
        CONTROL_FLOW_SAMPLES.len(),
        STRING_ENCODING_SAMPLES.len(),
        DEAD_CODE_SAMPLES.len(),
        VARIABLE_RENAMING_SAMPLES.len(),
        CODE_BLOAT_SAMPLES.len(),
        CONSTANT_RESTORATION_SAMPLES.len(),
        API_HIDING_SAMPLES.len(),
        DYNAMIC_INVOCATION_SAMPLES.len(),
        MIXED_SAMPLES.len(),
    )
}
