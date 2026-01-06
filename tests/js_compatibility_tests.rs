/// JavaScript Compatibility Tests
/// Tests M2 milestone requirements for AI-Centric Execution Refresh

#[cfg(test)]
mod tests {
    use browerai::dom::ResourceLimits;
    use browerai::{JsParser, JsSandbox};

    #[test]
    fn test_es_modules_import_detection() {
        let parser = JsParser::new();
        // ES modules will fail to parse - this is expected
        let result = parser.parse("import { foo } from 'bar';");
        // Without enforcement, it logs warning but may still error in parse
        // With Boa, this actually fails to parse
        assert!(result.is_err());
    }

    #[test]
    fn test_es_modules_export_detection() {
        let parser = JsParser::new();
        // ES modules will fail to parse - this is expected
        let result = parser.parse("export default foo;");
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_import_detection() {
        let parser = JsParser::new();
        // Dynamic import looks like a function call and may parse
        let result = parser.parse("const module = import('./module.js');");
        // This actually parses as it looks like a function call to Boa
        assert!(result.is_ok());
    }

    #[test]
    fn test_top_level_await_detection() {
        let parser = JsParser::new();
        // Top-level await will fail to parse - this is expected
        let result = parser.parse("const data = await fetch('/api');");
        assert!(result.is_err());
    }

    #[test]
    fn test_compatibility_enforcement_mode() {
        let mut parser = JsParser::new();
        parser.set_enforce_compatibility(true);
        assert!(parser.is_enforcing_compatibility());

        // Should fail with enforcement enabled
        let result = parser.parse("import foo from 'bar';");
        assert!(result.is_err());
    }

    #[test]
    fn test_compatible_code_passes() {
        let parser = JsParser::new();
        let code = r#"
            const foo = 42;
            function bar() {
                return foo * 2;
            }
            bar();
        "#;
        let result = parser.parse(code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_async_function_supported() {
        let parser = JsParser::new();
        let code = r#"
            async function fetchData() {
                const data = await Promise.resolve(42);
                return data;
            }
        "#;
        let result = parser.parse(code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_arrow_functions_supported() {
        let parser = JsParser::new();
        let code = "const add = (a, b) => a + b;";
        let result = parser.parse(code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_classes_supported() {
        let parser = JsParser::new();
        let code = r#"
            class Person {
                constructor(name) {
                    this.name = name;
                }
                greet() {
                    return `Hello, ${this.name}`;
                }
            }
        "#;
        let result = parser.parse(code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_default_limits() {
        let sandbox = JsSandbox::with_defaults();
        let stats = sandbox.get_stats();
        assert_eq!(stats.elapsed_ms, 0);
    }

    #[test]
    fn test_sandbox_custom_limits() {
        let limits = ResourceLimits {
            max_execution_time_ms: 1000,
            max_memory_bytes: 10 * 1024 * 1024,
            max_call_depth: 50,
            max_operations: 100_000,
        };
        let sandbox = JsSandbox::new(limits);
        let stats = sandbox.get_stats();
        assert_eq!(stats.elapsed_ms, 0);
    }

    #[test]
    fn test_sandbox_simple_execution() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.execute("2 + 2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_function_execution() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.execute(
            r#"
            function add(a, b) {
                return a + b;
            }
            add(2, 3)
        "#,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_variable_declaration() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.execute("const x = 42; x * 2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_compatibility_no_enforcement_by_default() {
        let parser = JsParser::new();
        assert!(!parser.is_enforcing_compatibility());
    }

    #[test]
    fn test_parser_handles_empty_code() {
        let parser = JsParser::new();
        let result = parser.parse("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parser_handles_comments() {
        let parser = JsParser::new();
        let code = r#"
            // Single line comment
            const x = 42;
            /* Multi-line
               comment */
            const y = 84;
        "#;
        let result = parser.parse(code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parser_handles_template_literals() {
        let parser = JsParser::new();
        let code = r#"
            const name = "World";
            const greeting = `Hello, ${name}!`;
        "#;
        let result = parser.parse(code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parser_handles_destructuring() {
        let parser = JsParser::new();
        let code = r#"
            const obj = { a: 1, b: 2 };
            const { a, b } = obj;
            const arr = [1, 2, 3];
            const [first, second] = arr;
        "#;
        let result = parser.parse(code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parser_handles_spread_operator() {
        let parser = JsParser::new();
        let code = r#"
            const arr1 = [1, 2, 3];
            const arr2 = [...arr1, 4, 5];
            const obj1 = { a: 1 };
            const obj2 = { ...obj1, b: 2 };
        "#;
        let result = parser.parse(code);
        assert!(result.is_ok());
    }
}
