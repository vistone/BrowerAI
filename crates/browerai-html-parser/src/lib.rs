use anyhow::Result;
use html5ever::parse_document;
use html5ever::tendril::TendrilSink;
use markup5ever_rcdom::{Handle, NodeData, RcDom};
use std::default::Default;
use std::io::Cursor;

/// HTML parser with AI enhancement capabilities
pub struct HtmlParser {
    // AI hooks removed; placeholder left for future reintroduction
}

impl HtmlParser {
    /// Create a new HTML parser
    pub fn new() -> Self {
        Self {}
    }

    /// Parse HTML content into a DOM tree
    pub fn parse(&self, html: &str) -> Result<RcDom> {
        let input = Cursor::new(html.as_bytes());
        let dom = parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut input.clone())?;

        log::info!("Successfully parsed HTML document");

        Ok(dom)
    }

    /// Extract text content from the DOM
    pub fn extract_text(&self, dom: &RcDom) -> String {
        let mut text = String::new();
        self.walk_tree(&dom.document, &mut text);
        text
    }

    /// Walk the DOM tree and collect text
    fn walk_tree(&self, handle: &Handle, text: &mut String) {
        let node = handle;

        if let NodeData::Text { ref contents } = node.data {
            text.push_str(&contents.borrow());
        }

        for child in node.children.borrow().iter() {
            self.walk_tree(child, text);
        }
    }

    /// Enable or disable AI enhancement (placeholder)
    #[allow(dead_code)]
    pub fn set_ai_enabled(&mut self, _enabled: bool) {}

    /// Check if AI enhancement is enabled (placeholder)
    #[allow(dead_code)]
    pub fn is_ai_enabled(&self) -> bool {
        false
    }
}

impl Default for HtmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_html() {
        let parser = HtmlParser::new();
        let html = "<html><body><h1>Hello, World!</h1></body></html>";
        let result = parser.parse(html);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_text() {
        let parser = HtmlParser::new();
        let html = "<html><body><p>Hello</p><p>World</p></body></html>";
        let dom = parser.parse(html).unwrap();
        let text = parser.extract_text(&dom);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_parse_malformed_html() {
        let parser = HtmlParser::new();
        let html = "<div><p>Unclosed paragraph<div>Nested</div>";
        let result = parser.parse(html);
        assert!(result.is_ok()); // html5ever is forgiving
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn parse_doesnt_crash(html in ".*") {
            let parser = HtmlParser::new();
            let _ = parser.parse(&html);
            // Should never panic, even with random input
        }

        #[test]
        fn parse_is_deterministic(html in ".*") {
            let parser = HtmlParser::new();
            let result1 = parser.parse(&html);
            let result2 = parser.parse(&html);
            prop_assert_eq!(result1.is_ok(), result2.is_ok());
        }

        #[test]
        fn parse_empty_or_whitespace_succeeds(s in r"[ \t\n\r]*") {
            let parser = HtmlParser::new();
            let result = parser.parse(&s);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn parse_simple_tags(tag in "[a-z]{1,10}", content in "[ -~]{0,100}") {
            let parser = HtmlParser::new();
            let html = format!("<{0}>{1}</{0}>", tag, content);
            let result = parser.parse(&html);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn extract_text_never_panics(html in ".*") {
            let parser = HtmlParser::new();
            if let Ok(dom) = parser.parse(&html) {
                let _ = parser.extract_text(&dom);
            }
        }
    }
}
