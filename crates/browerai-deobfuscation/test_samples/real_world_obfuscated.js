// Real-world obfuscated JavaScript samples for testing
// Source: Common obfuscation techniques found in the wild

// Sample 1: Packer obfuscation (Dean Edwards style)
eval(function(p,a,c,k,e,d){e=function(c){return c};if(!''.replace(/^/,String)){while(c--){d[c]=k[c]||c}k=[function(e){return d[e]}];e=function(){return'\\w+'};c=1};while(c--){if(k[c]){p=p.replace(new RegExp('\\b'+e(c)+'\\b','g'),k[c])}}return p}('0.1("2 3")',4,4,'console|log|Hello|World'.split('|'),0,{}))

// Sample 2: JSFuck style (limited character set)
[][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]+(!![]+[])[+[]]+(!![]+[])[!+[]+!+[]+!+[]]+(!![]+[])[+!+[]]]

// Sample 3: String array rotation with self-defending
var _0x4e8f=['log','Hello\x20World','test','length'];(function(_0x2d8f05,_0x4e8f71){var _0x4b291f=function(_0x32719f){while(--_0x32719f){_0x2d8f05['push'](_0x2d8f05['shift']());}};_0x4b291f(++_0x4e8f71);}(_0x4e8f,0x1b6));var _0x4b29=function(_0x2d8f05,_0x4e8f71){_0x2d8f05=_0x2d8f05-0x0;var _0x4b291f=_0x4e8f[_0x2d8f05];return _0x4b291f;};console[_0x4b29('0x0')](_0x4b29('0x1'));

// Sample 4: Control flow flattening
var a = 1;
switch(a) {
    case 1:
        console.log('one');
        a = 2;
    case 2:
        console.log('two');
        a = 3;
    default:
        console.log('done');
}

// Sample 5: Hex and Unicode escape sequences
var _0x1='\x48\x65\x6c\x6c\x6f';var _0x2='\u0057\u006f\u0072\u006c\u0064';console.log(_0x1+' '+_0x2);

// Sample 6: Base64 + eval injection
var encodedPayload='Y29uc29sZS5sb2coJ0Rhbmdlcm91cyBjb2RlIGV4ZWN1dGVkJyk=';eval(atob(encodedPayload));

// Sample 7: fromCharCode obfuscation
var msg=String['fromCharCode'](72,101,108,108,111,32,87,111,114,108,100);console['log'](msg);

// Sample 8: Dead code injection
function legitimate(){return 42;}
function unused1(){console.log('never called 1');}
function unused2(){console.log('never called 2');}
var result=legitimate();

// Sample 9: Number obfuscation
var a=0x10;var b=0o20;var c=0b1010;var sum=a+b+c;

// Sample 10: Property access obfuscation
var obj={method:function(){return'result'}};var key='method';var result=obj[key]();

// Sample 11: Complex nested obfuscation
var _0xabc=['test','\x6c\x6f\x67'];
(function(){
    var _0x1=String['\x66\x72\x6f\x6d\x43\x68\x61\x72\x43\x6f\x64\x65'](72,101,108,108,111);
    var _0x2=atob('V29ybGQ=');
    console[_0xabc[1]](_0x1+' '+_0x2);
})();

// Sample 12: Proxy trap and self-defending
var handler={get:function(obj,prop){return prop in obj?obj[prop]:'default';}};
var p=new Proxy({},handler);
if(typeof window!=='undefined'){
    setInterval(function(){if(new Date()%2===0){debugger;}},1000);
}

// Sample 13: Condition inversion
var x=10;if(!(x<5)){console.log('x is not less than 5');}

// Sample 14: Loop obfuscation with break conditions
var i=0;while(i<10){i++;if(i===5)continue;if(i===8)break;console.log(i);}

// Sample 15: Mixed encoding techniques
var s1='He'+'llo';
var s2=String.fromCharCode(87,111,114,108,100);
var s3=unescape('%57%6F%72%6C%64');
var s4=atob('SGVsbG8=');
console.log(s1,s2,s3,s4);
