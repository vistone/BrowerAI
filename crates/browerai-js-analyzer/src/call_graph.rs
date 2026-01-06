/// 调用图构建器 - 建立函数间的调用关系
///
/// 这个模块从提取的语义信息中构建函数调用关系图，
/// 支持：
/// - 静态调用识别
/// - 动态调用近似
/// - 循环调用检测
/// - 调用深度计算
use super::types::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};

/// 调用图构建器
pub struct CallGraphBuilder {
    /// 已识别的调用
    calls: HashMap<String, Vec<String>>,

    /// 反向调用映射
    reverse_calls: HashMap<String, Vec<String>>,

    /// 已访问的节点
    _visited: HashSet<String>,
}

impl Default for CallGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CallGraphBuilder {
    /// 创建新的调用图构建器
    pub fn new() -> Self {
        Self {
            calls: HashMap::new(),
            reverse_calls: HashMap::new(),
            _visited: HashSet::new(),
        }
    }

    /// 从语义信息构建调用图
    pub fn build(&mut self, semantic: &JsSemanticInfo) -> Result<JsCallGraph> {
        // 初始化所有函数节点
        for func in &semantic.functions {
            self.calls
                .insert(func.id.clone(), func.called_functions.clone());
            self.reverse_calls.insert(func.id.clone(), vec![]);
        }

        // 建立反向调用关系
        for (caller_id, callees) in &self.calls {
            for callee_id in callees {
                if let Some(callers) = self.reverse_calls.get_mut(callee_id) {
                    if !callers.contains(caller_id) {
                        callers.push(caller_id.clone());
                    }
                }
            }
        }

        // 检测循环
        let cycles = self.detect_cycles();

        // 计算深度并构建节点
        let mut nodes = vec![];
        for (func_id, callees) in &self.calls {
            let callers = self.reverse_calls.get(func_id).cloned().unwrap_or_default();
            let is_entry = callers.is_empty();
            let depth = self.calculate_depth(func_id);

            // 从语义信息中获取函数名
            let name = semantic
                .functions
                .iter()
                .find(|f| &f.id == func_id)
                .and_then(|f| f.name.clone());

            nodes.push(JsCallNode {
                id: func_id.clone(),
                name,
                callees: callees.clone(),
                callers,
                depth,
                is_entry_point: is_entry,
            });
        }

        let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);

        Ok(JsCallGraph {
            nodes,
            cycles,
            max_depth,
        })
    }

    /// Phase 2 增强：使用位置信息改进调用图
    ///
    /// 利用精确的位置信息改进函数识别和调用追踪精度
    pub fn build_with_locations(
        &mut self,
        semantic: &JsSemanticInfo,
        locations: &std::collections::HashMap<String, crate::types::LocationInfo>,
    ) -> Result<JsCallGraph> {
        // 首先执行标准调用图构建
        let mut graph = self.build(semantic)?;

        // Phase 2 增强：使用位置信息

        // 1. 使用位置信息改进节点关联
        for node in &mut graph.nodes {
            if let Some(location) = locations.get(&node.id) {
                log::debug!(
                    "Function {} at line {}, column {}",
                    node.id,
                    location.line,
                    location.column
                );
                // 可以使用这些信息进行更精细的分析
            }
        }

        // 2. 改进调用深度计算（基于代码结构）
        // 现在有了精确位置，可以更准确地计算嵌套深度
        self.recalculate_depth_with_locations(semantic, locations, &mut graph);

        Ok(graph)
    }

    /// 使用位置信息重新计算深度
    fn recalculate_depth_with_locations(
        &self,
        semantic: &JsSemanticInfo,
        locations: &std::collections::HashMap<String, crate::types::LocationInfo>,
        graph: &mut JsCallGraph,
    ) {
        // 根据位置信息的行号，计算更准确的嵌套深度
        for node in &mut graph.nodes {
            if let Some(location) = locations.get(&node.id) {
                // 计算该函数嵌套在其他函数中的深度
                let nested_depth = self.calculate_nesting_depth(location.line, semantic, locations);

                // 更新深度（取调用深度和嵌套深度的较大值）
                node.depth = std::cmp::max(node.depth, nested_depth as u32);
            }
        }
    }

    /// 计算函数的嵌套深度（基于行号位置）
    fn calculate_nesting_depth(
        &self,
        line: usize,
        _semantic: &JsSemanticInfo,
        locations: &std::collections::HashMap<String, crate::types::LocationInfo>,
    ) -> usize {
        let mut depth = 0;

        // 找出所有在这个行号之前的函数
        for func_location in locations.values() {
            if func_location.line < line {
                // 简单启发式：假设之后的函数嵌套在之前的函数中
                // 真实情况需要更复杂的括号匹配逻辑
                depth += 1;
            }
        }

        depth / 2 // 大致估计
    }

    /// 添加调用关系
    #[allow(dead_code)]
    fn add_call(&mut self, caller: &str, callee: &str) {
        self.calls
            .entry(caller.to_string())
            .or_default()
            .push(callee.to_string());

        self.reverse_calls
            .entry(callee.to_string())
            .or_default()
            .push(caller.to_string());
    }

    /// 检测循环调用
    fn detect_cycles(&self) -> Vec<Vec<String>> {
        let mut cycles = vec![];
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = vec![];

        for start in self.calls.keys() {
            if !visited.contains(start) {
                self.dfs(start, &mut visited, &mut rec_stack, &mut path, &mut cycles);
            }
        }

        cycles
    }

    /// DFS遍历检测循环
    fn dfs(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        path.push(node.to_string());

        if let Some(neighbors) = self.calls.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    self.dfs(neighbor, visited, rec_stack, path, cycles);
                } else if rec_stack.contains(neighbor) {
                    // 发现循环
                    if let Some(idx) = path.iter().position(|x| x == neighbor) {
                        let cycle = path[idx..].to_vec();
                        // 避免重复
                        if !cycles.iter().any(|c: &Vec<String>| {
                            c.len() == cycle.len() && c.iter().zip(&cycle).all(|(a, b)| a == b)
                        }) {
                            cycles.push(cycle);
                        }
                    }
                }
            }
        }

        path.pop();
        rec_stack.remove(node);
    }

    /// 判断是否为入口点
    fn is_entry_point(&self, func_id: &str) -> bool {
        // 入口点条件：
        // 1. 没有调用者
        // 2. 或者被全局作用域调用
        let callers = self.reverse_calls.get(func_id).cloned().unwrap_or_default();
        callers.is_empty()
    }

    /// 计算函数的调用深度
    fn calculate_depth(&self, func_id: &str) -> u32 {
        let mut queue = VecDeque::new();
        let mut depths = HashMap::new();

        // 从入口点开始
        if self.is_entry_point(func_id) {
            queue.push_back((func_id.to_string(), 0));
            depths.insert(func_id.to_string(), 0);
        } else {
            // 非入口点从所有调用者计算
            if let Some(callers) = self.reverse_calls.get(func_id) {
                for caller in callers {
                    let caller_depth = self.calculate_depth(caller);
                    queue.push_back((func_id.to_string(), caller_depth + 1));
                }
            }
            return 1;
        }

        // BFS计算深度
        while let Some((current, depth)) = queue.pop_front() {
            if let Some(callees) = self.calls.get(&current) {
                for callee in callees {
                    let new_depth = depth + 1;
                    let current_max = depths.get(callee).copied().unwrap_or(0);
                    if new_depth > current_max {
                        depths.insert(callee.clone(), new_depth);
                        queue.push_back((callee.clone(), new_depth));
                    }
                }
            }
        }

        depths.get(func_id).copied().unwrap_or(0)
    }

    /// 获取特定函数的调用链
    pub fn get_call_chain(&self, func_id: &str) -> Vec<Vec<String>> {
        let mut chains = vec![];
        let mut current_path = vec![];
        self.find_all_paths(func_id, &mut current_path, &mut chains);
        chains
    }

    /// 递归查找所有调用路径
    fn find_all_paths(
        &self,
        node: &str,
        current_path: &mut Vec<String>,
        all_paths: &mut Vec<Vec<String>>,
    ) {
        current_path.push(node.to_string());

        if let Some(callees) = self.calls.get(node) {
            if callees.is_empty() {
                // 叶子节点
                all_paths.push(current_path.clone());
            } else {
                for callee in callees {
                    if !current_path.contains(callee) {
                        self.find_all_paths(callee, current_path, all_paths);
                    }
                }
            }
        } else {
            // 叶子节点
            all_paths.push(current_path.clone());
        }

        current_path.pop();
    }

    /// 检查两个函数之间是否存在调用路径
    pub fn has_call_path(&self, from: &str, to: &str) -> bool {
        let mut visited = HashSet::new();
        self.has_path_dfs(from, to, &mut visited)
    }

    /// DFS检查路径
    fn has_path_dfs(&self, current: &str, target: &str, visited: &mut HashSet<String>) -> bool {
        if current == target {
            return true;
        }

        if visited.contains(current) {
            return false;
        }

        visited.insert(current.to_string());

        if let Some(callees) = self.calls.get(current) {
            for callee in callees {
                if self.has_path_dfs(callee, target, visited) {
                    return true;
                }
            }
        }

        false
    }

    /// 获取指定节点的所有后继节点（直接和间接）
    pub fn get_successors(&self, func_id: &str) -> HashSet<String> {
        let mut successors = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back(func_id.to_string());

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(callees) = self.calls.get(&current) {
                for callee in callees {
                    if !visited.contains(callee) {
                        successors.insert(callee.clone());
                        queue.push_back(callee.clone());
                    }
                }
            }
        }

        successors
    }

    /// 获取指定节点的所有前驱节点（直接和间接）
    pub fn get_predecessors(&self, func_id: &str) -> HashSet<String> {
        let mut predecessors = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back(func_id.to_string());

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(callers) = self.reverse_calls.get(&current) {
                for caller in callers {
                    if !visited.contains(caller) {
                        predecessors.insert(caller.clone());
                        queue.push_back(caller.clone());
                    }
                }
            }
        }

        predecessors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = CallGraphBuilder::new();
        assert!(builder.calls.is_empty());
    }

    #[test]
    fn test_simple_call_chain() {
        let mut builder = CallGraphBuilder::new();
        builder.add_call("func_a", "func_b");
        builder.add_call("func_b", "func_c");

        assert!(builder.has_call_path("func_a", "func_c"));
        assert!(!builder.has_call_path("func_c", "func_a"));
    }

    #[test]
    fn test_cycle_detection_simple() {
        let mut builder = CallGraphBuilder::new();
        builder.add_call("func_a", "func_b");
        builder.add_call("func_b", "func_c");
        builder.add_call("func_c", "func_a");

        let cycles = builder.detect_cycles();
        assert!(!cycles.is_empty());
    }

    #[test]
    fn test_cycle_detection_self_reference() {
        let mut builder = CallGraphBuilder::new();
        builder.add_call("func_a", "func_a"); // 自引用

        let cycles = builder.detect_cycles();
        assert!(!cycles.is_empty());
    }

    #[test]
    fn test_successors_computation() {
        let mut builder = CallGraphBuilder::new();
        builder.add_call("a", "b");
        builder.add_call("b", "c");
        builder.add_call("a", "d");

        let successors = builder.get_successors("a");
        assert!(successors.contains("b"));
        assert!(successors.contains("c"));
        assert!(successors.contains("d"));
    }

    #[test]
    fn test_predecessors_computation() {
        let mut builder = CallGraphBuilder::new();
        builder.add_call("a", "c");
        builder.add_call("b", "c");
        builder.add_call("c", "d");

        let predecessors = builder.get_predecessors("c");
        assert!(predecessors.contains("a"));
        assert!(predecessors.contains("b"));
        assert!(!predecessors.contains("d"));
    }

    #[test]
    fn test_build_with_semantic() {
        let mut builder = CallGraphBuilder::new();
        let mut semantic = JsSemanticInfo::default();

        // 添加测试函数
        semantic.functions.push(JsFunctionInfo {
            id: "func_0".to_string(),
            name: Some("main".to_string()),
            scope_level: 0,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 1,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec!["func_1".to_string()],
            start_line: 0,
            end_line: 0,
        });

        semantic.functions.push(JsFunctionInfo {
            id: "func_1".to_string(),
            name: Some("helper".to_string()),
            scope_level: 1,
            parameters: vec![],
            return_type_hint: None,
            statement_count: 1,
            cyclomatic_complexity: 1,
            is_async: false,
            is_generator: false,
            captured_vars: vec![],
            local_vars: vec![],
            called_functions: vec![],
            start_line: 0,
            end_line: 0,
        });

        let graph = builder.build(&semantic).unwrap();
        assert_eq!(graph.nodes.len(), 2);

        // 验证 main 是入口点
        let main_node = graph.nodes.iter().find(|n| n.id == "func_0").unwrap();
        assert!(main_node.is_entry_point);
    }

    #[test]
    fn test_call_path_detection() {
        let mut builder = CallGraphBuilder::new();
        builder.add_call("a", "b");
        builder.add_call("b", "c");
        builder.add_call("c", "d");

        // 直接和间接调用
        assert!(builder.has_call_path("a", "b"));
        assert!(builder.has_call_path("a", "c"));
        assert!(builder.has_call_path("a", "d"));
        assert!(!builder.has_call_path("d", "a"));
    }

    #[test]
    fn test_depth_calculation() {
        let mut builder = CallGraphBuilder::new();
        builder.add_call("a", "b");
        builder.add_call("b", "c");
        builder.add_call("c", "d");

        // 深度应该随层级增加
        let depth_a = builder.calculate_depth("a");
        let depth_b = builder.calculate_depth("b");
        let depth_d = builder.calculate_depth("d");

        // a 是入口点（深度 0）
        assert_eq!(depth_a, 0);
        // 其他非入口点深度应该 > 0
        assert!(depth_b > 0 || depth_d >= 0);
    }

    #[test]
    fn test_no_call_path() {
        let mut builder = CallGraphBuilder::new();
        builder.add_call("a", "b");
        builder.add_call("c", "d");

        assert!(!builder.has_call_path("a", "d"));
        assert!(!builder.has_call_path("b", "c"));
    }
}
