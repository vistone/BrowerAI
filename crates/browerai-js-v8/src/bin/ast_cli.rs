use anyhow::{Context, Result};
use browerai_js_v8::V8JsParser;
use serde_json::json;
use std::env;
use std::fs;
use std::io::{self, Read};

fn read_input() -> Result<String> {
    let args: Vec<String> = env::args().collect();
    // Usage: ast_cli [file|-]
    if args.len() > 1 {
        let path = &args[1];
        if path == "-" {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            return Ok(buf);
        }
        let data =
            fs::read_to_string(path).with_context(|| format!("Failed to read file: {}", path))?;
        Ok(data)
    } else {
        // No args: read stdin
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        Ok(buf)
    }
}

fn main() -> Result<()> {
    let js = read_input()?;
    let mut parser = V8JsParser::new()?;

    let mut compiled = false;
    let is_valid = match parser.parse(&js) {
        Ok(_ast) => {
            compiled = true;
            true
        }
        Err(_) => false,
    };

    let out = json!({
        "used_v8": true,
        "is_valid": is_valid,
        "compiled": compiled,
        "source_length": js.len(),
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
