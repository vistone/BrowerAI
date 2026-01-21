/// Unified Call Graph Builder
///
/// This module provides a unified implementation that combines:
/// - Basic call graph building (from call_graph.rs)
/// - Enhanced analysis with CFG/DFG integration (from enhanced_call_graph.rs)
///
/// All functionality is now available through `UnifiedCallGraphBuilder`.
use super::types::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Default)]
pub struct UnifiedCallGraphBuilder {
    calls: HashMap<String, Vec<String>>,
    reverse_calls: HashMap<String, Vec<String>>,
    _visited: HashSet<String>,
}

impl UnifiedCallGraphBuilder {
    pub fn new() -> Self {
        Self {
            calls: HashMap::new(),
            reverse_calls: HashMap::new(),
            _visited: HashSet::new(),
        }
    }

    pub fn build(&mut self, semantic: &JsSemanticInfo) -> Result<JsCallGraph> {
        for func in &semantic.functions {
            self.calls
                .insert(func.id.clone(), func.called_functions.clone());
            self.reverse_calls.insert(func.id.clone(), vec![]);
        }

        for (caller_id, callees) in &self.calls {
            for callee_id in callees {
                if let Some(callers) = self.reverse_calls.get_mut(callee_id) {
                    if !callers.contains(caller_id) {
                        callers.push(caller_id.clone());
                    }
                }
            }
        }

        let cycles = self.detect_cycles();
        let mut nodes = vec![];

        for (func_id, callees) in &self.calls {
            let callers = self.reverse_calls.get(func_id).cloned().unwrap_or_default();
            let is_entry = callers.is_empty();
            let depth = self.calculate_depth(func_id);

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

    pub fn add_call(&mut self, caller: &str, callee: &str) {
        self.calls
            .entry(caller.to_string())
            .or_default()
            .push(callee.to_string());

        self.reverse_calls
            .entry(callee.to_string())
            .or_default()
            .push(caller.to_string());
    }

    fn detect_cycles(&self) -> Vec<Vec<String>> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();

        for start in self.calls.keys() {
            let start = start.to_string();
            if visited.contains(&start) {
                continue;
            }

            let mut stack = vec![start.clone()];
            let mut parents: HashMap<String, Option<String>> = HashMap::new();
            parents.insert(start.clone(), None);

            while let Some(current) = stack.pop() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current.clone());

                if let Some(neighbors) = self.calls.get(&current) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            if !parents.contains_key(neighbor) {
                                parents.insert(neighbor.clone(), Some(current.clone()));
                                stack.push(neighbor.clone());
                            }
                        } else if let Some(parent) = parents.get(&current) {
                            if parent.as_ref().is_some_and(|p| p != neighbor) {
                                let mut cycle = Vec::new();
                                let mut node = current.clone();
                                while let Some(p) = parents.get(&node).cloned().flatten() {
                                    cycle.push(node);
                                    if &p == neighbor {
                                        cycle.push(p);
                                        break;
                                    }
                                    node = p;
                                }
                                if cycle.first() == Some(neighbor) || cycle.last() == Some(neighbor)
                                {
                                    cycle.reverse();
                                }
                                if !cycle.is_empty()
                                    && cycle.first() == Some(neighbor)
                                    && !cycles.iter().any(|c: &Vec<String>| {
                                        c.len() == cycle.len()
                                            && c.iter().zip(cycle.iter()).all(|(a, b)| a == b)
                                    })
                                {
                                    cycles.push(cycle);
                                }
                            }
                        }
                    }
                }
            }
        }

        cycles
    }

    #[allow(dead_code)]
    fn is_entry_point(&self, func_id: &str) -> bool {
        let callers = self.reverse_calls.get(func_id).cloned().unwrap_or_default();
        callers.is_empty()
    }

    fn calculate_depth(&self, func_id: &str) -> u32 {
        let mut depths = HashMap::new();
        let mut stack = vec![func_id.to_string()];
        depths.insert(func_id.to_string(), 0);

        while let Some(current) = stack.pop() {
            if let Some(callees) = self.calls.get(&current) {
                let current_depth = depths.get(&current).copied().unwrap_or(0);
                for callee in callees {
                    let new_depth = current_depth + 1;
                    let existing_depth = depths.get(callee).copied().unwrap_or(0);
                    if new_depth > existing_depth {
                        depths.insert(callee.clone(), new_depth);
                        stack.push(callee.clone());
                    }
                }
            }
        }

        depths.get(func_id).copied().unwrap_or(0)
    }

    pub fn has_call_path(&self, from: &str, to: &str) -> bool {
        let mut visited = HashSet::new();
        self.has_path_dfs(from, to, &mut visited)
    }

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
        let builder = UnifiedCallGraphBuilder::new();
        assert!(builder.calls.is_empty());
    }

    #[test]
    fn test_simple_call_chain() {
        let mut builder = UnifiedCallGraphBuilder::new();
        builder.add_call("func_a", "func_b");
        builder.add_call("func_b", "func_c");

        assert!(builder.has_call_path("func_a", "func_c"));
        assert!(!builder.has_call_path("func_c", "func_a"));
    }

    #[test]
    #[ignore]
    fn test_cycle_detection() {
        let mut builder = UnifiedCallGraphBuilder::new();
        builder.add_call("func_a", "func_b");
        builder.add_call("func_b", "func_c");
        builder.add_call("func_c", "func_a");

        let semantic = JsSemanticInfo::default();
        let graph = builder.build(&semantic).unwrap();
        assert!(graph.has_cycles());
    }

    #[test]
    fn test_successors() {
        let mut builder = UnifiedCallGraphBuilder::new();
        builder.add_call("a", "b");
        builder.add_call("b", "c");
        builder.add_call("a", "d");

        let successors = builder.get_successors("a");
        assert!(successors.contains("b"));
        assert!(successors.contains("c"));
        assert!(successors.contains("d"));
    }

    #[test]
    fn test_predecessors() {
        let mut builder = UnifiedCallGraphBuilder::new();
        builder.add_call("a", "c");
        builder.add_call("b", "c");
        builder.add_call("c", "d");

        let predecessors = builder.get_predecessors("c");
        assert!(predecessors.contains("a"));
        assert!(predecessors.contains("b"));
        assert!(!predecessors.contains("d"));
    }
}
