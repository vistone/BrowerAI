#![no_main]

use libfuzzer_sys::fuzz_target;
use browerai_html_parser::HtmlParser;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let parser = HtmlParser::new();
        let _ = parser.parse(s);
        let _ = parser.extract_text(s);
    }
});
