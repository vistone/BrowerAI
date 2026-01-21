// WebAssembly 分析器 - 二进制分析和功能提取
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Cursor, Seek};

const WASM_MAGIC: u32 = 0x6D736100;
const WASM_VERSION: u32 = 0x00000001;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleInfo {
    pub name: Option<String>,
    pub url: String,
    pub size_bytes: usize,
    pub imports: Vec<WasmImport>,
    pub exports: Vec<WasmExport>,
    pub functions: Vec<WasmFunction>,
    pub memory: WasmMemory,
    pub tables: Vec<WasmTable>,
    pub globals: Vec<WasmGlobal>,
    pub custom_sections: HashMap<String, Vec<u8>>,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmImport {
    pub module: String,
    pub name: String,
    pub kind: ImportKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmExport {
    pub name: String,
    pub kind: ExportKind,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportKind {
    Function {
        type_index: u32,
    },
    Table {
        elem_type: u8,
        flags: u32,
        initial: u32,
        maximum: Option<u32>,
    },
    Memory {
        flags: u32,
        initial: u32,
        maximum: Option<u32>,
    },
    Global {
        content_type: ValueType,
        mutable: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportKind {
    Function,
    Table,
    Memory,
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub params: Vec<ValueType>,
    pub results: Vec<ValueType>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValueType {
    I32,
    I64,
    F32,
    F64,
    V128,
    FuncRef,
    ExternRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmFunction {
    pub index: u32,
    pub name: Option<String>,
    pub signature: FunctionSignature,
    pub locals: Vec<ValueType>,
    pub instructions_count: usize,
    pub complexity_score: f64,
    pub is_exported: bool,
    pub calls: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmMemory {
    pub initial_pages: u32,
    pub maximum_pages: Option<u32>,
    pub is_shared: bool,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmTable {
    pub initial_size: u32,
    pub maximum_size: Option<u32>,
    pub element_type: ValueType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmGlobal {
    pub value_type: ValueType,
    pub is_mutable: bool,
    pub init_expr: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmDataSegment {
    pub index: u32,
    pub offset: Vec<u8>,
    pub data: Vec<u8>,
}

pub struct WasmAnalyzer {
    enable_decompilation: bool,
}

impl Default for WasmAnalyzer {
    fn default() -> Self {
        Self::new(false)
    }
}

impl WasmAnalyzer {
    pub fn new(enable_decompilation: bool) -> Self {
        Self {
            enable_decompilation,
        }
    }

    pub fn analyze(&self, wasm_bytes: &[u8], url: &str) -> Result<WasmModuleInfo> {
        log::info!("开始分析 WASM 模块: {} ({} bytes)", url, wasm_bytes.len());

        if wasm_bytes.len() < 8 {
            return Err(anyhow::anyhow!(
                "Invalid WASM: too small ({} bytes)",
                wasm_bytes.len()
            ));
        }

        let mut reader = Cursor::new(wasm_bytes.to_vec());

        let magic = read_u32(&mut reader)?;
        if magic != WASM_MAGIC {
            return Err(anyhow::anyhow!(
                "Invalid WASM magic number: {:x}, expected {:x}",
                magic,
                WASM_MAGIC
            ));
        }

        let version = read_u32(&mut reader)?;
        if version != WASM_VERSION {
            return Err(anyhow::anyhow!("Unsupported WASM version: {}", version));
        }

        let mut module_info = WasmModuleInfo {
            name: None,
            url: url.to_string(),
            size_bytes: wasm_bytes.len(),
            imports: Vec::new(),
            exports: Vec::new(),
            functions: Vec::new(),
            memory: WasmMemory {
                initial_pages: 0,
                maximum_pages: None,
                is_shared: false,
                bytes: None,
            },
            tables: Vec::new(),
            globals: Vec::new(),
            custom_sections: HashMap::new(),
            version,
        };

        let mut function_type_indices: Vec<u32> = Vec::new();
        let mut type_section: Vec<FunctionSignature> = Vec::new();

        while reader.position() < wasm_bytes.len() as u64 {
            let section_id = read_u8(&mut reader)?;
            let section_size = read_varuint(&mut reader)?;

            let section_start = reader.position();
            match section_id {
                0 => {
                    let name_section = read_name_section_data(&mut reader, section_size as usize)?;
                    module_info
                        .custom_sections
                        .insert("name".to_string(), name_section);
                }
                1 => {
                    type_section = read_type_section(&mut reader)?;
                }
                2 => {
                    function_type_indices = read_function_section(&mut reader)?;
                }
                3 => {
                    module_info.tables = read_table_section(&mut reader)?;
                }
                4 => {
                    module_info.memory = read_memory_section(&mut reader)?;
                }
                5 => {
                    module_info.globals = read_global_section(&mut reader)?;
                }
                6 => {
                    module_info.exports = read_export_section(&mut reader)?;
                }
                7 => {
                    module_info.imports = read_import_section(&mut reader)?;
                }
                _ => {
                    reader.seek(std::io::SeekFrom::Current(section_size as i64))?;
                }
            }

            let current = reader.position();
            if current > section_start + section_size as u64 {
                return Err(anyhow::anyhow!("Section parsing overflow"));
            }
        }

        module_info.functions = build_functions(&function_type_indices, &type_section);

        if let Some(name_section) = module_info.custom_sections.get("name").cloned() {
            apply_name_section(&mut module_info, &name_section);
        }

        module_info
            .functions
            .iter_mut()
            .enumerate()
            .for_each(|(idx, func)| {
                func.index = idx as u32;
                func.is_exported = module_info
                    .exports
                    .iter()
                    .any(|e| matches!(e.kind, ExportKind::Function) && e.index == func.index);
            });

        log::info!(
            "WASM 分析完成: {} 个函数, {} 个导出",
            module_info.functions.len(),
            module_info.exports.len()
        );

        Ok(module_info)
    }

    pub fn decompile_to_pseudocode(&self, module: &WasmModuleInfo) -> Result<String> {
        if !self.enable_decompilation {
            return Err(anyhow::anyhow!("Decompilation not enabled"));
        }

        let mut pseudocode = String::new();

        pseudocode.push_str(&format!("// WASM Module: {}\n", module.url));
        pseudocode.push_str(&format!("// Size: {} bytes\n", module.size_bytes));
        pseudocode.push_str(&format!("// Version: {}\n\n", module.version));

        pseudocode.push_str("// === Imports ===\n");
        for import in &module.imports {
            match &import.kind {
                ImportKind::Function { type_index } => {
                    pseudocode.push_str(&format!(
                        "//   import {} {} :: (type {})\n",
                        import.module, import.name, type_index
                    ));
                }
                _ => {
                    pseudocode
                        .push_str(&format!("//   import {} {}\n", import.module, import.name));
                }
            }
        }

        pseudocode.push_str("\n// === Exports ===\n");
        for export in &module.exports {
            pseudocode.push_str(&format!("//   {}: {:?}\n", export.name, export.kind));
        }

        pseudocode.push_str("\n// === Functions ===\n");
        for func in &module.functions {
            pseudocode.push_str(&format!("\n{}\n", self.generate_function_pseudocode(func)));
        }

        Ok(pseudocode)
    }

    fn generate_function_pseudocode(&self, func: &WasmFunction) -> String {
        let name = if let Some(ref n) = func.name {
            n.clone()
        } else {
            format!("func_{}", func.index)
        };

        let params: Vec<String> = func
            .signature
            .params
            .iter()
            .enumerate()
            .map(|(i, t)| format!("p{}: {:?}", i, t))
            .collect();

        let returns: Vec<String> = func
            .signature
            .results
            .iter()
            .map(|t| format!("{:?}", t))
            .collect();

        let return_str = if returns.is_empty() {
            "void".to_string()
        } else {
            returns.join(", ")
        };

        let complexity = if func.instructions_count > 100 {
            "high".to_string()
        } else if func.instructions_count > 50 {
            "medium".to_string()
        } else {
            "low".to_string()
        };

        format!(
            "function {}({}) -> {} {{\n  // {} instructions, complexity: {}, exported: {}\n  // TODO: full decompilation requires wasmparser library\n}}",
            name,
            params.join(", "),
            return_str,
            func.instructions_count,
            complexity,
            func.is_exported
        )
    }

    pub fn build_call_graph(&self, module: &WasmModuleInfo) -> WasmCallGraph {
        let mut graph = WasmCallGraph {
            nodes: HashMap::new(),
            edges: vec![],
        };

        for func in &module.functions {
            graph.nodes.insert(func.index, func.clone());
        }

        for func in &module.functions {
            for &called_index in &func.calls {
                graph.edges.push((func.index, called_index));
            }
        }

        graph
    }

    pub fn infer_function_purpose(&self, func: &WasmFunction) -> String {
        if let Some(name) = &func.name {
            let name_lower = name.to_lowercase();
            if name_lower.contains("add")
                || name_lower.contains("sum")
                || name_lower.contains("plus")
            {
                return "Mathematical addition".to_string();
            }
            if name_lower.contains("sub") || name_lower.contains("minus") {
                return "Mathematical subtraction".to_string();
            }
            if name_lower.contains("mul") {
                return "Mathematical multiplication".to_string();
            }
            if name_lower.contains("div") {
                return "Mathematical division".to_string();
            }
            if name_lower.contains("malloc") || name_lower.contains("alloc") {
                return "Memory allocation".to_string();
            }
            if name_lower.contains("free") || name_lower.contains("release") {
                return "Memory deallocation".to_string();
            }
            if name_lower.contains("init") {
                return "Initialization".to_string();
            }
            if name_lower.contains("encode") || name_lower.contains("decode") {
                return "Encoding/decoding".to_string();
            }
            if name_lower.contains("hash") || name_lower.contains("digest") {
                return "Cryptographic hash".to_string();
            }
            if name_lower.contains("encrypt") || name_lower.contains("decrypt") {
                return "Encryption/decryption".to_string();
            }
            if name_lower.contains("get") || name_lower.contains("query") {
                return "Getter/query function".to_string();
            }
            if name_lower.contains("set") || name_lower.contains("update") {
                return "Setter/update function".to_string();
            }
        }

        if func.signature.params.is_empty() && func.signature.results.is_empty() {
            return "Procedure with side effects".to_string();
        }

        if func.signature.params.is_empty() && func.signature.results.len() == 1 {
            return "Getter or constant function".to_string();
        }

        if func.complexity_score > 10.0 {
            return "Complex computation or algorithm".to_string();
        }

        "Unknown purpose".to_string()
    }
}

fn read_u8<R: std::io::Read>(reader: &mut R) -> Result<u8> {
    let mut byte = [0u8];
    reader.read_exact(&mut byte)?;
    Ok(byte[0])
}

fn read_u32<R: std::io::Read>(reader: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_varuint<R: std::io::Read>(reader: &mut R) -> Result<u64> {
    let mut result = 0u64;
    let mut shift = 0;
    loop {
        let byte = read_u8(reader)?;
        result |= ((byte as u64) & 0x7F) << shift;
        if (byte & 0x80) == 0 {
            break;
        }
        shift += 7;
    }
    Ok(result)
}

fn read_value_type<R: std::io::Read>(reader: &mut R) -> Result<ValueType> {
    let byte = read_u8(reader)?;
    match byte {
        0x7F => Ok(ValueType::I32),
        0x7E => Ok(ValueType::I64),
        0x7D => Ok(ValueType::F32),
        0x7C => Ok(ValueType::F64),
        0x7B => Ok(ValueType::V128),
        0x70 => Ok(ValueType::FuncRef),
        0x6F => Ok(ValueType::ExternRef),
        _ => Err(anyhow::anyhow!("Unknown value type: {}", byte)),
    }
}

fn read_type_section<R: std::io::Read>(reader: &mut R) -> Result<Vec<FunctionSignature>> {
    let count = read_varuint(reader)? as usize;
    let mut types = Vec::with_capacity(count);

    for _ in 0..count {
        let byte = read_u8(reader)?;
        if byte != 0x60 {
            return Err(anyhow::anyhow!(
                "Expected function type (0x60), got {:x}",
                byte
            ));
        }

        let param_count = read_varuint(reader)? as usize;
        let mut params = Vec::with_capacity(param_count);
        for _ in 0..param_count {
            params.push(read_value_type(reader)?);
        }

        let result_count = read_varuint(reader)? as usize;
        let mut results = Vec::with_capacity(result_count);
        for _ in 0..result_count {
            results.push(read_value_type(reader)?);
        }

        types.push(FunctionSignature { params, results });
    }

    Ok(types)
}

fn read_function_section<R: std::io::Read>(reader: &mut R) -> Result<Vec<u32>> {
    let count = read_varuint(reader)? as usize;
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        indices.push(read_varuint(reader)? as u32);
    }
    Ok(indices)
}

fn read_table_section<R: std::io::Read>(reader: &mut R) -> Result<Vec<WasmTable>> {
    let count = read_varuint(reader)? as usize;
    let mut tables = Vec::with_capacity(count);

    for _ in 0..count {
        let _elem_type = read_u8(reader)?;
        let flags = read_varuint(reader)? as u32;
        let initial = read_varuint(reader)? as u32;
        let maximum = if flags & 0x1 != 0 {
            Some(read_varuint(reader)? as u32)
        } else {
            None
        };

        tables.push(WasmTable {
            initial_size: initial,
            maximum_size: maximum,
            element_type: ValueType::FuncRef,
        });
    }

    Ok(tables)
}

fn read_memory_section<R: std::io::Read>(reader: &mut R) -> Result<WasmMemory> {
    let count = read_varuint(reader)? as usize;
    if count > 1 {
        return Err(anyhow::anyhow!("Multiple memory sections not supported"));
    }

    if count == 0 {
        return Ok(WasmMemory {
            initial_pages: 0,
            maximum_pages: None,
            is_shared: false,
            bytes: None,
        });
    }

    let flags = read_varuint(reader)?;
    let initial = read_varuint(reader)? as u32;
    let maximum = if flags & 0x1 != 0 {
        Some(read_varuint(reader)? as u32)
    } else {
        None
    };
    let is_shared = (flags & 0x2) != 0;

    Ok(WasmMemory {
        initial_pages: initial,
        maximum_pages: maximum,
        is_shared,
        bytes: None,
    })
}

fn read_global_section<R: std::io::Read>(reader: &mut R) -> Result<Vec<WasmGlobal>> {
    let count = read_varuint(reader)? as usize;
    let mut globals = Vec::with_capacity(count);

    for _ in 0..count {
        let value_type = read_value_type(reader)?;
        let mutable = read_u8(reader)? == 0x01;

        let mut init_expr = Vec::new();
        loop {
            let opcode = read_u8(reader)?;
            init_expr.push(opcode);
            if opcode == 0x0B {
                break;
            }
        }

        globals.push(WasmGlobal {
            value_type,
            is_mutable: mutable,
            init_expr,
        });
    }

    Ok(globals)
}

fn read_export_section<R: std::io::Read>(reader: &mut R) -> Result<Vec<WasmExport>> {
    let count = read_varuint(reader)? as usize;
    let mut exports = Vec::with_capacity(count);

    for _ in 0..count {
        let name = read_name(reader)?;
        let kind_byte = read_u8(reader)?;
        let index = read_varuint(reader)? as u32;

        let kind = match kind_byte {
            0x00 => ExportKind::Function,
            0x01 => ExportKind::Table,
            0x02 => ExportKind::Memory,
            0x03 => ExportKind::Global,
            _ => ExportKind::Function,
        };

        exports.push(WasmExport { name, kind, index });
    }

    Ok(exports)
}

fn read_import_section<R: std::io::Read>(reader: &mut R) -> Result<Vec<WasmImport>> {
    let count = read_varuint(reader)? as usize;
    let mut imports = Vec::with_capacity(count);

    for _ in 0..count {
        let module = read_name(reader)?;
        let name = read_name(reader)?;
        let kind_byte = read_u8(reader)?;

        let kind = match kind_byte {
            0x00 => {
                let type_index = read_varuint(reader)? as u32;
                ImportKind::Function { type_index }
            }
            0x01 => ImportKind::Table {
                elem_type: 0x70,
                flags: 0,
                initial: 0,
                maximum: None,
            },
            0x02 => ImportKind::Memory {
                flags: 0,
                initial: 0,
                maximum: None,
            },
            0x03 => {
                let content_type = read_value_type(reader)?;
                let mutable = read_u8(reader)? == 0x01;
                ImportKind::Global {
                    content_type,
                    mutable,
                }
            }
            _ => ImportKind::Function { type_index: 0 },
        };

        imports.push(WasmImport { module, name, kind });
    }

    Ok(imports)
}

fn read_name_section_data<R: std::io::Read>(
    reader: &mut R,
    _section_size: usize,
) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    Ok(data)
}

fn read_name<R: std::io::Read>(reader: &mut R) -> Result<String> {
    let len = read_varuint(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).context("Invalid UTF-8 in name")
}

fn build_functions(function_indices: &[u32], types: &[FunctionSignature]) -> Vec<WasmFunction> {
    function_indices
        .iter()
        .enumerate()
        .map(|(idx, &type_idx)| {
            let signature =
                types
                    .get(type_idx as usize)
                    .cloned()
                    .unwrap_or_else(|| FunctionSignature {
                        params: vec![ValueType::I32],
                        results: vec![ValueType::I32],
                    });

            let instructions_count = (idx + 1) * 10;
            let complexity_score = (instructions_count as f64) / 10.0;

            WasmFunction {
                index: idx as u32,
                name: None,
                signature,
                locals: Vec::new(),
                instructions_count,
                complexity_score,
                is_exported: false,
                calls: Vec::new(),
            }
        })
        .collect()
}

fn apply_name_section(_module: &mut WasmModuleInfo, _name_data: &[u8]) {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmCallGraph {
    pub nodes: HashMap<u32, WasmFunction>,
    pub edges: Vec<(u32, u32)>,
}

impl WasmCallGraph {
    pub fn find_entry_points(&self) -> Vec<u32> {
        let called: std::collections::HashSet<_> =
            self.edges.iter().map(|(_, callee)| *callee).collect();

        self.nodes
            .keys()
            .filter(|&idx| !called.contains(idx))
            .copied()
            .collect()
    }

    pub fn find_leaf_functions(&self) -> Vec<u32> {
        let callers: std::collections::HashSet<_> =
            self.edges.iter().map(|(caller, _)| *caller).collect();

        self.nodes
            .keys()
            .filter(|&idx| !callers.contains(idx))
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_analyzer_creation() {
        let analyzer = WasmAnalyzer::new(true);
        assert_eq!(analyzer.enable_decompilation, true);

        let default_analyzer = WasmAnalyzer::default();
        assert_eq!(default_analyzer.enable_decompilation, false);
    }

    #[test]
    fn test_wasm_magic_check() {
        let analyzer = WasmAnalyzer::default();

        let valid_wasm = create_minimal_wasm();
        let result = analyzer.analyze(&valid_wasm, "test.wasm");
        assert!(result.is_ok());

        let invalid_wasm = b"invalid";
        let result = analyzer.analyze(invalid_wasm, "invalid.wasm");
        assert!(result.is_err());
    }

    #[test]
    fn test_function_purpose_inference() {
        let analyzer = WasmAnalyzer::default();

        let func = WasmFunction {
            index: 0,
            name: Some("add_numbers".to_string()),
            signature: FunctionSignature {
                params: vec![ValueType::I32, ValueType::I32],
                results: vec![ValueType::I32],
            },
            locals: vec![],
            instructions_count: 10,
            complexity_score: 2.0,
            is_exported: true,
            calls: vec![],
        };

        let purpose = analyzer.infer_function_purpose(&func);
        assert!(purpose.contains("addition"));
    }

    #[test]
    fn test_call_graph_entry_points() {
        let mut graph = WasmCallGraph {
            nodes: HashMap::new(),
            edges: vec![(0, 1), (1, 2)],
        };

        for i in 0..3 {
            graph.nodes.insert(
                i,
                WasmFunction {
                    index: i,
                    name: Some(format!("func_{}", i)),
                    signature: FunctionSignature {
                        params: vec![],
                        results: vec![],
                    },
                    locals: vec![],
                    instructions_count: 10,
                    complexity_score: 1.0,
                    is_exported: false,
                    calls: vec![],
                },
            );
        }

        let entry_points = graph.find_entry_points();
        assert_eq!(entry_points.len(), 1);
        assert!(entry_points.contains(&0));
    }

    fn create_minimal_wasm() -> Vec<u8> {
        vec![
            0x00, 0x61, 0x73, 0x6D, // WASM magic
            0x01, 0x00, 0x00, 0x00, // WASM version
            0x01, 0x04, // Type section
            0x01, // 1 type
            0x60, // func type
            0x00, 0x00, // no params, no results
            0x02, 0x07, // Function section
            0x01, // 1 function
            0x00, // type index 0
            0x07, 0x0E, // Export section
            0x01, // 1 export
            0x04, 0x6D, 0x61, 0x69, 0x6E, // "main"
            0x00, // func
            0x00, // index 0
            0x0A, 0x09, // Code section
            0x01, // 1 function
            0x07, // function body size
            0x00, // local count
            0x0B, // end
        ]
    }
}
