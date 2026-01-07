#![no_main]

use libfuzzer_sys::fuzz_target;
use browerai_css_parser::CssParser;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let mut parser = CssParser::new();
        let _ = parser.parse(s);
        let _ = parser.validate(s);
    }
});
